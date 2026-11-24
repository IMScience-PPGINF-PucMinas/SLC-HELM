"""Trains SCH_HELM2 with VGG+NetVLAD superpixel features on AID (graphs from build_dataset.sh).

    python train.py --seed 1
    python train.py --seed 1 --aggregation mean   # ablation: mean-pooled VGG features
    python train.py --seed 1 --readout lse --config configs/trial_53.json  # configuration found by search.py

Short-epoch + EMA protocol (epoch = 1/10 of the augmented training set, patiences counted in those
epochs, weight averages tracked in parallel and the checkpoint chosen on validation among the model
and its averages):

    python train.py --seed 1 --epochs 3000 --epoch-fraction 0.1 --plateau-patience 30 --patience 100 \\
        --ema 0.99 0.999 0.9997 --test-every 10

A snapshot of the full training state is written every --snapshot-every epochs and the run resumes
from it when restarted with the same arguments.
"""
import argparse
import json
import os

import mlflow
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import RandomSampler
from torch_geometric.loader import DataLoader

from config import MLFLOW_URI, N_NODES, NUM_CLASSES, WEIGHTS_DIR
from dataset import graphs, init_worker
from model import SCH_HELM2


CONFIG_KEYS = {"n_nodes": "nodes"}  # keys of the search configurations that are named differently here


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="JSON with hyperparameters (keys of search.py, e.g. configs/trial_53.json) that "
                   "replace the defaults; arguments given explicitly still win. The run name gets the file name")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32,
                   help="64 in RESISC45; with 600x600 images 64 takes ~18 GB of GPU memory, plus fragmentation")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=3e-6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--embedding-size", type=int, default=64)
    p.add_argument("--percentile", type=float, default=50)
    p.add_argument("--aggregation", choices=["netvlad", "mean"], default="netvlad")
    p.add_argument("--num-clusters", type=int, default=16)
    p.add_argument("--readout", choices=["att", "lse"], default="lse",
                   help="scale weighting of the readout: att, learned score per scale; lse, softmax of the logsumexp score of each scale (the paper's runs)")
    p.add_argument("--threshold", choices=["batch", "graph"], default="batch",
                   help="pooling percentile over all edges of the batch (original) or over the edges of each graph")
    p.add_argument("--mgp", choices=["branches", "seq_sh"], default="branches",
                   help="branches: parallel spatial/hierarchical Graphers + FFN; seq_sh: single branch, spatial then hierarchical")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--nodes", type=int, default=N_NODES,
                   help="base number of superpixels of the graphs (build_dataset.py --nodes)")
    p.add_argument("--patience", type=int, default=30, help="early stopping patience (epochs)")
    p.add_argument("--plateau-patience", type=int, default=10, help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--epoch-fraction", type=float, default=1.0,
                   help="fraction of the training set drawn (without replacement) for each epoch")
    p.add_argument("--ema", type=float, nargs="*", default=[],
                   help="EMA decays (per step) tracked alongside the model; the saved checkpoint is the best on validation")
    p.add_argument("--test-every", type=int, default=1, help="epochs between test-set evaluations of the model")
    p.add_argument("--snapshot-every", type=int, default=10, help="epochs between resumable snapshots (0: off)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-workers", type=int, default=None, help="workers of the val/test loaders (default: --workers)")
    p.add_argument("--protocol", default=None,
                   help="train/val/test protocol of split_protocol.py (splits/<name>.npz, e.g. tr50: 50%% of the images "
                   "of each class for training and validation, the rest for test); default: the torchgeo splits")
    p.add_argument("--experiment", default=None, help="MLflow experiment (default: named by --nodes and --protocol)")
    args = p.parse_args()
    if args.config:
        with open(args.config) as f:
            config = {CONFIG_KEYS.get(k, k): v for k, v in json.load(f).items()}
        unknown = set(config) - set(vars(args))
        if unknown:
            p.error(f"{args.config}: unknown hyperparameters {sorted(unknown)}")
        p.set_defaults(**config)
        args = p.parse_args()
    if args.experiment is None:
        args.experiment = f"AID - SCH_HELM2 - VGG NetVLAD - {args.nodes} nodes"
        if args.protocol:
            args.experiment += f" - {args.protocol}"
    return args


def build_model(args):
    return SCH_HELM2(num_classes=NUM_CLASSES, embedding_size=args.embedding_size, dropout=args.dropout,
                     heads=args.heads, percentile=args.percentile, aggregation=args.aggregation,
                     num_clusters=args.num_clusters, readout=args.readout, mgp=args.mgp,
                     threshold=args.threshold)


def run_name(args):
    name = f"SCH_HELM2_{args.aggregation}_{args.readout}"
    if args.nodes != N_NODES:
        name += f"_n{args.nodes}"
    if args.protocol:
        name += f"_{args.protocol}"
    if args.config:
        name += "_" + os.path.splitext(os.path.basename(args.config))[0]
    if args.threshold == "graph":
        name += "_pg"
    if args.mgp != "branches":
        name += f"_{args.mgp}"
    if args.epoch_fraction != 1:
        name += f"_ep{args.epoch_fraction:g}"
    if args.ema:
        name += "_ema"
    return f"{name}_seed={args.seed}"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_epoch(model, loader, loss_fn, device, optimizer=None, on_step=None):
    """One pass over `loader`; trains if an optimizer is given (calling `on_step` after each update).
    Returns (mean loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if on_step is not None:
                    on_step()
            total_loss += loss.item() * batch.num_graphs
            correct += (pred.argmax(dim=1) == batch.y).sum().item()
            seen += batch.num_graphs
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate_many(nets, loader, loss_fn, device):
    """Evaluates several networks in one pass over `loader`. Returns {key: (mean loss, accuracy)}."""
    for net in nets.values():
        net.eval()
    totals = {key: [0.0, 0] for key in nets}
    seen = 0
    for batch in loader:
        batch = batch.to(device)
        for key, net in nets.items():
            pred = net(batch)
            totals[key][0] += loss_fn(pred, batch.y).item() * batch.num_graphs
            totals[key][1] += (pred.argmax(dim=1) == batch.y).sum().item()
        seen += batch.num_graphs
    return {key: (loss / seen, correct / seen) for key, (loss, correct) in totals.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = {split: graphs(split, args.nodes, args.protocol) for split in ["train", "val", "test"]}
    if len(datasets["val"]) == 0:
        raise SystemExit(f"protocol {args.protocol} has no validation images, which train.py needs")
    sampler_gen = None
    if args.epoch_fraction != 1:
        sampler_gen = torch.Generator().manual_seed(args.seed)
        num_samples = int(len(datasets["train"]) * args.epoch_fraction)
        train_kw = dict(sampler=RandomSampler(datasets["train"], num_samples=num_samples, generator=sampler_gen))
    else:
        train_kw = dict(shuffle=True)
    def workers(split):
        return args.workers if split == "train" or args.eval_workers is None else args.eval_workers

    loaders = {
        split: DataLoader(ds, batch_size=args.batch_size, num_workers=workers(split), persistent_workers=workers(split) > 0,
                          worker_init_fn=init_worker, **(train_kw if split == "train" else {}))
        for split, ds in datasets.items()
    }

    model = build_model(args).to(device)
    print(f"Trainable parameters: {count_parameters(model)}")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(WEIGHTS_DIR, run_name(args) + ".pth")
    snap_path = os.path.join(WEIGHTS_DIR, run_name(args) + ".snap.pt")

    # Weight averages (BatchNorm statistics included); "model" is the network being trained.
    emas = {f"ema{d:g}": AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(d), use_buffers=True) for d in args.ema}
    nets = {"model": model, **emas}

    def ckpt_path(key):  # without EMAs the model checkpoint keeps the original file name
        return save_path if not emas else os.path.join(WEIGHTS_DIR, f"{run_name(args)}__{key}.pth")

    def plain_state(key):  # state_dict loadable by build_model(args)
        net = nets[key].module if key in emas else nets[key]
        return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                 betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=args.plateau_patience,
                                                           min_lr=1e-6, factor=0.5)

    # best[key] = (val accuracy, epoch, state_dict); early stopping and the LR schedule follow the model itself
    best = {key: (0.0, -1, None) for key in nets}
    no_improve, start_epoch, run_id = 0, 0, None
    if args.snapshot_every and os.path.exists(snap_path):
        snap = torch.load(snap_path, map_location="cpu", weights_only=False)
        model.load_state_dict(snap["model"])
        for key, ema in emas.items():
            ema.load_state_dict(snap["emas"][key])
        optimizer.load_state_dict(snap["optimizer"])
        scheduler.load_state_dict(snap["scheduler"])
        best, no_improve, start_epoch, run_id = snap["best"], snap["no_improve"], snap["epoch"] + 1, snap["mlflow_run_id"]
        for key, (_, _, state) in best.items():  # checkpoints on disk match the snapshot
            if state is not None:
                torch.save(state, ckpt_path(key))
        torch.set_rng_state(snap["cpu_rng"])
        torch.cuda.set_rng_state_all(snap["cuda_rng"])
        np.random.set_state(snap["np_rng"])
        if sampler_gen is not None:
            sampler_gen.set_state(snap["sampler_rng"])
        print(f"Resuming from epoch {start_epoch + 1} ({snap_path})")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=run_name(args), run_id=run_id) as run:
        if run_id is None:
            mlflow.log_params(vars(args))
        for epoch in range(start_epoch, args.epochs):
            if no_improve >= args.patience:  # stopped right before an interruption
                break
            metrics = {}
            metrics["train_loss"], metrics["train_acc"] = run_epoch(
                model, loaders["train"], loss_fn, device, optimizer,
                on_step=lambda: [ema.update_parameters(model) for ema in emas.values()])
            for key, (loss, acc) in evaluate_many(nets, loaders["val"], loss_fn, device).items():
                suffix = "" if key == "model" else f"_{key}"
                metrics[f"val_loss{suffix}"], metrics[f"val_acc{suffix}"] = loss, acc
            if (epoch + 1) % args.test_every == 0:
                metrics["test_loss"], metrics["test_acc"] = run_epoch(model, loaders["test"], loss_fn, device)
            metrics["lr"] = optimizer.param_groups[0]["lr"]
            mlflow.log_metrics(metrics, step=epoch)
            print(f"Epoch {epoch + 1:3d} | " + " | ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

            for key in nets:
                acc = metrics["val_acc" if key == "model" else f"val_acc_{key}"]
                if acc > best[key][0]:
                    best[key] = (acc, epoch, plain_state(key))
                    torch.save(best[key][2], ckpt_path(key))
                    print(f"  new best val accuracy for {key}: {acc:.4f}, saved to {ckpt_path(key)}")
            no_improve = 0 if best["model"][1] == epoch else no_improve + 1
            stop = no_improve >= args.patience
            if not stop:
                scheduler.step(metrics["val_acc"])
            if args.snapshot_every and ((epoch + 1) % args.snapshot_every == 0 or stop):
                torch.save(dict(model=model.state_dict(), emas={k: e.state_dict() for k, e in emas.items()},
                                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), best=best,
                                no_improve=no_improve, epoch=epoch, mlflow_run_id=run.info.run_id,
                                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                                np_rng=np.random.get_state(),
                                sampler_rng=sampler_gen.get_state() if sampler_gen is not None else None),
                           snap_path + ".tmp")
                os.replace(snap_path + ".tmp", snap_path)
            if stop:
                print(f"Early stopping after {epoch + 1} epochs (best val accuracy {best['model'][0]:.4f})")
                break

        if emas:  # the saved checkpoint is the network (model or one of its averages) with the best validation accuracy
            chosen = max(best, key=lambda k: best[k][0])
            torch.save(best[chosen][2], save_path)
            for key, (acc, ep, _) in best.items():
                print(f"  {key:10s}: best val accuracy {acc:.4f} at epoch {ep + 1}")
                mlflow.log_metric(f"best_val_acc_{key}", acc)
            mlflow.log_param("checkpoint", chosen)
            print(f"Checkpoint: {chosen} (val accuracy {best[chosen][0]:.4f}, epoch {best[chosen][1] + 1}) -> {save_path}")
    if os.path.exists(snap_path):
        os.remove(snap_path)


if __name__ == "__main__":
    main()
