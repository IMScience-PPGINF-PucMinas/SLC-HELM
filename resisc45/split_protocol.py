"""Train/val/test protocols of RESISC45 other than the torchgeo split of data/images (60/20/20), such as the usual
20%/80% protocol: a stratified fraction of the images of each class (of all 31500) is the labelled set, the rest is
the test set, and a fraction of the labelled set is held out for validation (LR schedule, early stopping and
checkpoint choice of train.py).

    python split_protocol.py --train-percent 20                   # splits/tr20.npz: 126 train + 14 val + 560 test per class
    python split_protocol.py --train-percent 20 --val-fraction 0  # splits/tr20_noval.npz: 140 train + 560 test

No graph is built: the labelled images are drawn from the torchgeo train split, whose images already have the
superpixel maps of all 51 augmentations (data/processed and data/processed_nN), and every image has the map of its
unaugmented version (val/test graphs, and augmentation 0, the identity, of the train images). A uniform draw from
the torchgeo train images of a class is a uniform draw from all images of the class, since that split was drawn at
random. For a given --train-percent and --seed the labelled set (so the test set) does not depend on --val-fraction,
and a smaller --train-percent gives a subset of the labelled set of a larger one. The labelled images of each class
form folds of --val-fraction of them; the file holds fold 0 as validation (protocol_fold gives the others, used by
the confirmation of the hyperparameter search).

The protocol file stores, for each split, the source split (index into SOURCES) and the index of each image there;
dataset.ProtocolGraphs turns them into graphs (train: every augmentation; val and test: the unaugmented graph).
Needs only numpy and config.py.
"""
import argparse
import os

import numpy as np

from config import IMAGES_DIR, N_AUGMENTATIONS, N_NODES, NUM_CLASSES, ROOT, graph_path

SPLITS_DIR = os.path.join(ROOT, "splits")
SOURCES = ("train", "val", "test")  # torchgeo splits of data/images; codes are positions in this tuple
SPLITS = ("train", "val", "test")   # splits of a protocol


def protocol_path(name):
    return os.path.join(SPLITS_DIR, f"{name}.npz")


def load_protocol(name):
    """{split: (source codes, image indices in the source split)} of protocol `name`."""
    z = np.load(protocol_path(name))
    return {split: (z[f"{split}_source"], z[f"{split}_image"]) for split in SPLITS}


def graphs(name, split):
    """(source codes, graph indices in the source split) of the graphs of `split` of protocol `name`: every
    augmentation of the training images, the unaugmented graph of the others."""
    source, image = load_protocol(name)[split]
    augs = np.arange(N_AUGMENTATIONS) if split == "train" else np.zeros(1, np.int64)
    source, image, k = np.repeat(source, len(augs)), np.repeat(image.astype(np.int64), len(augs)), np.tile(augs, len(image))
    from_train = source == SOURCES.index("train")
    return source, np.where(from_train, image * N_AUGMENTATIONS + k, image)


def source_graphs(name):
    """{source split: sorted graph indices} of every graph of protocol `name`: what a copy of it must hold."""
    out = {s: [np.zeros(0, np.int64)] for s in SOURCES}
    for split in SPLITS:
        source, graph = graphs(name, split)
        for code, s in enumerate(SOURCES):
            out[s].append(graph[source == code])
    return {s: np.unique(np.concatenate(g)) for s, g in out.items()}


def make_protocol(train_percent, val_fraction, seed, fold=0):
    """Fields of the protocol file. The labelled images of each class are split into folds of val_fraction of them;
    `fold` is the one held out for validation (fold 0 in the protocol files, others in the search's confirmation)."""
    labels = [np.load(os.path.join(IMAGES_DIR, f"{s}_labels.npy")) for s in SOURCES]
    out = {split: [] for split in SPLITS}  # lists of (source code, image index) arrays
    for c in range(NUM_CLASSES):
        members = [np.flatnonzero(lab == c) for lab in labels]
        n_labelled = round(sum(len(m) for m in members) * train_percent / 100)
        n_val = round(n_labelled * val_fraction)
        if n_labelled > len(members[0]):
            raise SystemExit(f"class {c}: {n_labelled} labelled images, but only {len(members[0])} in the train split")
        if (fold + 1) * n_val > n_labelled:
            raise SystemExit(f"fold {fold}: only {n_labelled // max(n_val, 1)} folds of {n_val} labelled images per class")
        labelled = np.random.default_rng([seed, c]).permutation(members[0])[:n_labelled]
        is_val = np.zeros(n_labelled, bool)
        is_val[fold * n_val:(fold + 1) * n_val] = True
        out["val"].append(np.stack([np.zeros(n_val, np.int64), labelled[is_val]], 1))
        out["train"].append(np.stack([np.zeros(n_labelled - n_val, np.int64), labelled[~is_val]], 1))
        rest = [np.setdiff1d(members[0], labelled), members[1], members[2]]
        out["test"].append(np.concatenate([np.stack([np.full(len(r), s), r], 1) for s, r in enumerate(rest)]))
    fields = {}
    for split, parts in out.items():
        pairs = np.concatenate(parts)
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        fields[f"{split}_source"], fields[f"{split}_image"] = pairs[:, 0].astype(np.uint8), pairs[:, 1].astype(np.int32)
    return fields


def protocol_fold(name, fold):
    """{split: (source codes, image indices)} of protocol `name` with validation fold `fold` (fold 0: the file)."""
    z = np.load(protocol_path(name))
    fields = make_protocol(float(z["train_percent"]), float(z["val_fraction"]), int(z["seed"]), fold)
    if fold == 0:
        assert all(np.array_equal(fields[k], z[k]) for k in fields), f"{protocol_path(name)} differs from its parameters"
    return {split: (fields[f"{split}_source"], fields[f"{split}_image"]) for split in SPLITS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-percent", type=float, default=20, help="labelled images per class (%% of all of its images)")
    p.add_argument("--val-fraction", type=float, default=0.1, help="fraction of the labelled images held out for validation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", help="protocol name (default: tr<percent>, with _noval if --val-fraction is 0)")
    p.add_argument("--nodes", type=int, nargs="*", default=[N_NODES, 300],
                   help="base numbers of superpixels whose graph files are checked")
    p.add_argument("--force", action="store_true", help="overwrite an existing protocol file")
    args = p.parse_args()
    name = args.name or f"tr{args.train_percent:g}" + ("_noval" if args.val_fraction == 0 else "")
    path = protocol_path(name)
    if os.path.exists(path) and not args.force:
        raise SystemExit(f"{path} exists (--force to overwrite)")

    fields = make_protocol(args.train_percent, args.val_fraction, args.seed)
    labels = [np.load(os.path.join(IMAGES_DIR, f"{s}_labels.npy")) for s in SOURCES]
    seen = set()
    for split in SPLITS:
        source, image = fields[f"{split}_source"], fields[f"{split}_image"]
        pairs = set(zip(source.tolist(), image.tolist()))
        assert len(pairs) == len(source) and not pairs & seen, f"{split}: repeated images"
        seen |= pairs
        per_class = np.bincount([labels[s][i] for s, i in zip(source, image)], minlength=NUM_CLASSES)
        from_source = np.bincount(source, minlength=len(SOURCES))
        print(f"{split}: {len(source)} images ({per_class.min()}-{per_class.max()} per class; from "
              + ", ".join(f"{s} {n}" for s, n in zip(SOURCES, from_source)) + ")")
    assert len(seen) == sum(len(lab) for lab in labels), "images missing from the protocol"

    os.makedirs(SPLITS_DIR, exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, train_percent=args.train_percent, val_fraction=args.val_fraction, seed=args.seed, **fields)
    os.replace(tmp, path)
    print(f"saved {path}")

    for nodes in args.nodes:  # every graph of the protocol must already exist
        counts, missing = {}, {}
        for split in SPLITS:
            source, graph = graphs(name, split)
            counts[split] = len(graph)
            missing[split] = sum(not os.path.exists(graph_path(SOURCES[s], g, nodes)) for s, g in zip(source, graph))
        print(f"graphs with {nodes} superpixels: " + ", ".join(f"{s} {counts[s]} ({missing[s]} missing)" for s in SPLITS))


if __name__ == "__main__":
    main()
