"""SCH_HELM2 (GATv2 + dynamic single-link hierarchy) with learned superpixel features.

Node features come from the frozen first layers of an ImageNet CNN (before its first max-pooling: VGG16 conv1_1..relu1_2,
or ResNet50 conv1+bn1+relu, whose stride-2 map is interpolated back to the image size) whose per-pixel descriptors are
aggregated per superpixel by a NetVLAD layer and projected by an MLP, as in Lézoray, "Learning to merge superpixels"
(NN variant: no normalization).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components
from torch_geometric.nn import BatchNorm, GATv2Conv, GraphSizeNorm, Linear
from torch_geometric.nn import global_add_pool as gdp, global_max_pool as gmp, global_mean_pool as gap
from torch_geometric.utils import coalesce, degree, scatter, softmax, to_scipy_sparse_matrix
from torchvision.models import ResNet50_Weights, VGG16_Weights, resnet50, vgg16

NUM_SCALES = 4  # original graph + 3 poolings


# ----------------------------------------------------------------------------------------
# Superpixel features
# ----------------------------------------------------------------------------------------

BACKBONES = ("vgg16", "resnet50")
UPSAMPLING = ("bilinear", "bicubic")


class PixelEncoder(nn.Module):
    """Frozen layers of an ImageNet CNN before its first max-pooling: [B, 3, H, W] uint8 -> [B, 64, H, W].

    vgg16:    conv1_1, ReLU, conv1_2, ReLU (full resolution).
    resnet50: conv1 (7x7, stride 2), bn1, ReLU -> [B, 64, H/2, W/2], interpolated (`upsample`: bilinear or bicubic)
              back to H x W, so each pixel of pix2node has a descriptor. bn1 keeps the ImageNet statistics: the
              layers stay in eval mode even when the model trains.
    """

    out_dim = 64

    def __init__(self, backbone="vgg16", upsample="bilinear", pretrained=True):
        super().__init__()
        if backbone == "vgg16":
            self.layers = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None).features[:4]
        elif backbone == "resnet50":
            net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            self.layers = nn.Sequential(net.conv1, net.bn1, net.relu)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        if upsample not in UPSAMPLING:
            raise ValueError(f"Unknown upsampling: {upsample}")
        self.upsample = upsample
        self.layers.requires_grad_(False)
        self.layers.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.layers.eval()  # frozen: BatchNorm (ResNet50) uses and keeps the ImageNet statistics
        return self

    @torch.no_grad()
    def forward(self, img):
        x = self.layers((img.float() / 255 - self.mean) / self.std)
        if x.shape[-2:] != img.shape[-2:]:
            x = F.interpolate(x, size=img.shape[-2:], mode=self.upsample, align_corners=False)
        return x


class NetVLAD(nn.Module):
    """NetVLAD over variable-size groups of descriptors: [P, C] -> [S, K*C].

    The per-group sums are batched matrix products: the descriptors are sorted by group and packed
    into zero-padded chunks of `chunk` rows. A 256x256 image has 65536 descriptors, and summing the
    K weighted copies of [P, C] by scatter (the EuroSAT version) took most of the training step.
    """

    def __init__(self, dim, num_clusters=16, normalize=False, chunk=64):
        super().__init__()
        self.num_clusters = num_clusters
        self.normalize = normalize
        self.chunk = chunk
        self.assign = nn.Linear(dim, num_clusters)  # the 1x1 convolution (a_k, b_k)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def forward(self, x, index, num_segments):
        if self.normalize:
            x = F.normalize(x, dim=1)
        x, mask, chunk_segment = self._chunks(x, index, num_segments)       # [N, L, C], [N, L, 1], [N]
        a = F.softmax(self.assign(x), dim=2) * mask                          # [N, L, K], 0 on padding rows
        # r_k = sum_p a_kp (x_p - c_k) = sum_p a_kp x_p - (sum_p a_kp) c_k
        weighted = x.new_zeros(num_segments, self.num_clusters, x.size(2)).index_add(
            0, chunk_segment, a.transpose(1, 2) @ x)                          # [S, K, C]
        mass = a.new_zeros(num_segments, self.num_clusters).index_add(0, chunk_segment, a.sum(dim=1))  # [S, K]
        v = weighted - mass.unsqueeze(-1) * self.centroids
        if self.normalize:
            v = F.normalize(F.normalize(v, dim=2).flatten(1), dim=1)
        return v.flatten(1)

    def _chunks(self, x, index, num_segments):
        """Rows of x grouped by segment in zero-padded chunks of self.chunk rows.

        Returns the chunks [N, L, C], their mask of real rows [N, L, 1] and the segment of each chunk [N].
        """
        size = self.chunk
        order = index.argsort()
        segment = index[order]
        counts = torch.bincount(index, minlength=num_segments)
        num_chunks = (counts + size - 1) // size
        first_row = (num_chunks.cumsum(0) - num_chunks) * size               # first padded row of each segment
        rank = torch.arange(index.numel(), device=x.device) - (counts.cumsum(0) - counts)[segment]
        rows = first_row[segment] + rank
        total = int(num_chunks.sum())
        padded = x.new_zeros(total * size, x.size(1))
        padded[rows] = x[order]
        mask = x.new_zeros(total * size, 1)
        mask[rows] = 1
        chunk_segment = torch.arange(num_segments, device=x.device).repeat_interleave(num_chunks)
        return padded.view(total, size, -1), mask.view(total, size, 1), chunk_segment


class SuperpixelEncoder(nn.Module):
    """Per-superpixel features from the image: [learned embedding || mean RGB || mean xy]."""

    def __init__(self, out_dim=64, aggregation="netvlad", num_clusters=16, hidden=256,
                 normalize=False, backbone="vgg16", upsample="bilinear", pretrained=True):
        super().__init__()
        self.pixels = PixelEncoder(backbone, upsample, pretrained)
        c = PixelEncoder.out_dim
        if aggregation == "netvlad":
            self.vlad = NetVLAD(c, num_clusters, normalize)
            in_dim = num_clusters * c
        elif aggregation == "mean":
            self.vlad = None
            in_dim = c
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.norm = BatchNorm(out_dim)
        self.out_dim = out_dim + 5

    def forward(self, img, pix2node, num_nodes):
        b, _, h, w = img.shape
        pix2node = pix2node.long()  # the dataset sends it as int32
        feats = self.pixels(img).permute(0, 2, 3, 1).reshape(b * h * w, -1)  # same order as pix2node
        if self.vlad is not None:
            v = self.vlad(feats, pix2node, num_nodes)
        else:
            v = scatter(feats, pix2node, dim=0, dim_size=num_nodes, reduce="mean")
        emb = self.norm(self.proj(v))

        rgb = img.float().div(255).permute(0, 2, 3, 1).reshape(b * h * w, 3)
        ys, xs = torch.meshgrid(torch.arange(h, device=img.device), torch.arange(w, device=img.device),
                                indexing="ij")
        xy = torch.stack([xs, ys], dim=-1).reshape(h * w, 2).float().div(max(h, w) - 1).repeat(b, 1)
        handcrafted = scatter(torch.cat([rgb, xy], dim=1), pix2node, dim=0, dim_size=num_nodes, reduce="mean")
        return torch.cat([emb, handcrafted], dim=1)


# ----------------------------------------------------------------------------------------
# Graph network
# ----------------------------------------------------------------------------------------

class SingleLinkPooling(nn.Module):
    """Single-link clustering pooling.

    Edges whose feature distance is below the given percentile of the edge distances are kept;
    connected components of the kept edges become the pooled nodes (mean of their features).
    The percentile is taken over all edges in the batch, or over the edges of each graph when
    per_graph=True (then the clusters of an image do not depend on the other images of its batch).
    Returns (pooled_x, pooled_edge_index, pooled_batch, cluster) where cluster maps each input
    node to its pooled node.
    """

    def __init__(self, distance_metric="euclidean", percentile=50, per_graph=False):
        super().__init__()
        self.distance_metric = distance_metric
        self.percentile = percentile
        self.per_graph = per_graph

    def forward(self, x, edge_index, batch):
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        active_edges = edge_index
        if edge_index.numel() > 0:
            distances = self._edge_distances(x, edge_index)
            if self.per_graph:
                edge_batch = batch[edge_index[0]]
                threshold = self._per_graph_threshold(distances, edge_batch)[edge_batch]
            else:
                k = max(1, int(distances.numel() * self.percentile / 100))
                threshold = torch.kthvalue(distances, k)[0]
            active_edges = edge_index[:, distances <= threshold]

        adj = to_scipy_sparse_matrix(active_edges.cpu(), num_nodes=x.size(0))
        _, cluster = connected_components(adj, directed=False, connection="weak")
        cluster = torch.as_tensor(cluster, dtype=torch.long, device=x.device)
        return self._aggregate(x, edge_index, batch, cluster)

    def _per_graph_threshold(self, distances, edge_batch):
        """k-th smallest edge distance of each graph, with the same k rule as the batch threshold."""
        num_edges = torch.bincount(edge_batch, minlength=int(edge_batch.max()) + 1)
        k = (num_edges.double() * self.percentile / 100).long().clamp(min=1)
        # sort by graph, then by distance inside each graph
        perm = distances.argsort()
        perm = perm[edge_batch[perm].argsort(stable=True)]
        start = num_edges.cumsum(0) - num_edges
        # graphs without edges get a dummy value that no edge reads
        return distances[perm][(start + k - 1).clamp(max=distances.numel() - 1)]

    def _edge_distances(self, x, edge_index):
        src, dst = x[edge_index[0]], x[edge_index[1]]
        if self.distance_metric == "euclidean":
            return torch.norm(src - dst, dim=1)
        if self.distance_metric == "cosine":
            return 1 - F.cosine_similarity(src, dst, dim=1)
        raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    @staticmethod
    def _aggregate(x, edge_index, batch, cluster):
        num_clusters = int(cluster.max()) + 1
        pooled_x = scatter(x, cluster, dim=0, dim_size=num_clusters, reduce="mean")

        src, dst = cluster[edge_index[0]], cluster[edge_index[1]]
        inter = src != dst
        pooled_edge_index = torch.stack([src[inter], dst[inter]])
        if pooled_edge_index.numel() > 0:
            pooled_edge_index = coalesce(pooled_edge_index, num_nodes=num_clusters)

        # All nodes of a cluster belong to the same graph.
        pooled_batch = scatter(batch, cluster, dim=0, dim_size=num_clusters, reduce="max")
        return pooled_x, pooled_edge_index, pooled_batch, cluster


class FFN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.linear1 = Linear(in_dim, hidden)
        self.linear2 = Linear(hidden, out_dim)
        self.bn1 = BatchNorm(hidden)
        self.bn2 = BatchNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.bn1(torch.relu(self.linear1(x))))
        return self.dropout(self.bn2(torch.relu(self.linear2(x))))


class Grapher(nn.Module):
    """Residual block of `num_convs` stacked GATv2 convolutions."""

    def __init__(self, dim, num_convs=1, dropout=0.2, heads=4):
        super().__init__()
        # concat=False averages the heads, keeping the residual shape.
        self.convs = nn.ModuleList([
            GATv2Conv(dim, dim, heads=heads, concat=False, dropout=dropout) for _ in range(num_convs)
        ])
        self.bn1 = BatchNorm(dim)
        self.bn2 = BatchNorm(dim)
        self.bn3 = BatchNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_in = Linear(dim, dim)
        self.linear_out = Linear(dim, dim)

    def forward(self, x, edge_index):
        x_in = self.linear_in(x)
        x = self.dropout(self.bn1(x_in))
        for i, conv in enumerate(self.convs):
            if i > 0:
                x = self.dropout(x)
            x = conv(x, edge_index)
        x = self.dropout(self.bn2(x))
        x = self.bn3(self.linear_out(x))
        return x + x_in


class SeqGrapherBlock(nn.Module):
    """Single-branch MGP block, spatial -> hierarchical.

    Each layer applies a spatial GATv2 and then a hierarchical GATv2 to the features the spatial one has
    just updated, each step as x <- x + dropout(ELU(BN(conv(x)))). The hierarchical conv has no
    self-loops and only updates the nodes with a parent or children (BN statistics over those nodes).
    The block ends with a non-residual FFN.
    """

    def __init__(self, dim, num_layers=2, dropout=0.2, heads=4):
        super().__init__()
        self.convs = nn.ModuleList([nn.ModuleDict({
            "s": GATv2Conv(dim, dim, heads=heads, concat=False, dropout=dropout),
            "h": GATv2Conv(dim, dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
        }) for _ in range(num_layers)])
        self.bns = nn.ModuleList([nn.ModuleDict({"s": BatchNorm(dim), "h": BatchNorm(dim)}) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFN(dim, dim, dropout=dropout)

    def forward(self, x, edge_s, edge_h):
        reached = (degree(edge_h[1], x.size(0)) > 0).nonzero().squeeze(1)
        for convs, bns in zip(self.convs, self.bns):
            x = x + self.dropout(F.elu(bns["s"](convs["s"](x, edge_s))))
            h = convs["h"](x, edge_h)[reached]
            x = x.index_add(0, reached, self.dropout(F.elu(bns["h"](h))))
        return self.ffn(x)


class ClassifierHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.linear1 = Linear(dim, dim * 2)
        self.linear2 = Linear(dim * 2, num_classes)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class SCH_HELM2(nn.Module):
    """Builds a 4-scale hierarchy by repeated single-link pooling.

    Nodes of all scales live in one tensor; `edge_s` holds the spatial (RAG) edges of every
    scale and `edge_h` the parent-child edges between consecutive scales. Blocks 2 and 3 (the MGP
    modules) process both relations and pool the newest scale:
      mgp='branches': spatial and hierarchical Graphers in parallel branches, fused by an FFN;
      mgp='seq_sh':   one branch, spatial then hierarchical conv in each layer (SeqGrapherBlock).
    threshold='batch' | 'graph': the pooling percentile is taken over the batch or over each graph.
    """

    def __init__(self, num_classes, embedding_size=64, dropout=0.2, heads=4, percentile=50,
                 aggregation="netvlad", num_clusters=16, pretrained=True, readout="att", mgp="branches",
                 threshold="batch", backbone="vgg16", upsample="bilinear"):
        super().__init__()
        d = embedding_size
        self.encoder = SuperpixelEncoder(d, aggregation, num_clusters, backbone=backbone, upsample=upsample,
                                         pretrained=pretrained)
        self.node_emb = Linear(self.encoder.out_dim, d)

        if threshold not in ("batch", "graph"):
            raise ValueError(f"Unknown threshold: {threshold}")
        self.poolings = nn.ModuleList([SingleLinkPooling(percentile=percentile, per_graph=threshold == "graph")
                                       for _ in range(NUM_SCALES - 1)])

        self.grapher_s1 = Grapher(d, num_convs=3, dropout=dropout, heads=heads)

        self.mgp = mgp
        if mgp == "branches":
            self.grapher_h1 = Grapher(d, dropout=dropout, heads=heads)
            self.grapher_s2 = Grapher(d, num_convs=3, dropout=dropout, heads=heads)
            self.ffn1 = FFN(2 * d, d, dropout=dropout)

            self.grapher_h2 = Grapher(d, dropout=dropout, heads=heads)
            self.grapher_s3 = Grapher(d, num_convs=3, dropout=dropout, heads=heads)
            self.ffn2 = FFN(2 * d, d, dropout=dropout)
        elif mgp == "seq_sh":
            self.block2 = SeqGrapherBlock(d, dropout=dropout, heads=heads)
            self.block3 = SeqGrapherBlock(d, dropout=dropout, heads=heads)
        else:
            raise ValueError(f"Unknown mgp: {mgp}")

        self.graph_size_norm = GraphSizeNorm()
        self.class1 = Linear(d, 256)
        # readout='att': learned score per scale on the mean of the scale;
        # readout='lse': softmax over scales of the logsumexp of the summed features (original version)
        self.readout = readout
        if readout == "att":
            self.scale_att = Linear(256, 1)
        elif readout != "lse":
            raise ValueError(f"Unknown readout: {readout}")
        self.head = ClassifierHead(256, num_classes)

    @staticmethod
    def _add_scale(pooling, scale, x, start, level_edges, level_batch, edge_s, edge_h, graph_index):
        """Pools the newest scale x[start:] and appends the result as scale `scale`."""
        end = x.size(0)
        pooled_x, pooled_edges, pooled_batch, cluster = pooling(x[start:], level_edges, level_batch)
        child = torch.arange(start, end, device=x.device)
        parent = cluster + end
        edge_h = torch.cat([edge_h, torch.stack([torch.cat([child, parent]), torch.cat([parent, child])])], dim=1)
        edge_s = torch.cat([edge_s, pooled_edges + end], dim=1)
        graph_index = torch.cat([graph_index, pooled_batch * NUM_SCALES + scale])
        return torch.cat([x, pooled_x]), end, pooled_edges, pooled_batch, edge_s, edge_h, graph_index

    def forward(self, data):
        edge_index, batch, num_graphs = data.edge_index, data.batch, data.num_graphs
        x = self.node_emb(self.encoder(data.img, data.pix2node, data.num_nodes))

        graph_index = batch * NUM_SCALES
        edge_s = edge_index
        edge_h = edge_index.new_empty(2, 0)
        level = (edge_index, batch)
        start = 0

        # Block 1
        x = self.grapher_s1(x, edge_index)
        x, start, *level, edge_s, edge_h, graph_index = self._add_scale(
            self.poolings[0], 1, x, start, *level, edge_s, edge_h, graph_index)

        # Blocks 2 and 3
        for scale in (2, 3):
            if self.mgp == "branches":
                grapher_h, grapher_s, ffn = ((self.grapher_h1, self.grapher_s2, self.ffn1),
                                             (self.grapher_h2, self.grapher_s3, self.ffn2))[scale - 2]
                x = ffn(torch.cat([grapher_s(x, edge_s), grapher_h(x, edge_h)], dim=1))
            else:
                x = (self.block2, self.block3)[scale - 2](x, edge_s, edge_h)
            x, start, *level, edge_s, edge_h, graph_index = self._add_scale(
                self.poolings[scale - 1], scale, x, start, *level, edge_s, edge_h, graph_index)

        # Readout: sum per (graph, scale), then softmax-weighted sum of the scales.
        num_slots = num_graphs * NUM_SCALES
        scale_to_graph = torch.arange(num_graphs, device=x.device).repeat_interleave(NUM_SCALES)
        x = self.class1(x)
        x_mean = gap(x, graph_index, size=num_slots)
        x = self.graph_size_norm(x, graph_index)
        x = gdp(x, graph_index, size=num_slots)                                     # [G*S, 256]

        if self.readout == "att":
            # Learned score per scale, computed on the mean so it does not depend on the scale's node count
            alpha = softmax(self.scale_att(x_mean), scale_to_graph)
        else:
            graph_max = gmp(x, scale_to_graph, size=num_graphs).max(dim=1, keepdim=True)[0]
            score = torch.exp(x - graph_max[scale_to_graph]).sum(dim=1, keepdim=True)   # [G*S, 1]
            alpha = score / gdp(score, scale_to_graph, size=num_graphs)[scale_to_graph]
        x = gdp(x * alpha, scale_to_graph, size=num_graphs)
        return self.head(x)
