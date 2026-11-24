"""CNN baseline: MobileNetV3-Small (torchvision, 1.0x) trained from scratch on RESISC45, with the protocol of the full
SCH_HELM2 runs (run_experiment.sh): the same splits, the same 51 training augmentations per image (the images the
graphs were built from: augmentations.augment with the seeds of build_dataset.py, and for the entries in
STORED_AUGMENTATIONS the image stored with the graph), all of them in every epoch, ReduceLROnPlateau patience 10,
early stopping patience 30 on the validation accuracy, batch 64, 256x256 input, Adam (betas 0.9/0.98, eps 1e-9) with
the learning rate, weight decay and label smoothing of the SCH_HELM2 runs (--config, default configs/trial_53.json;
its other keys are GNN hyperparameters and are ignored). No pretrained weights. After training, the checkpoint with
the best validation accuracy is evaluated on the test set.

    python cnn_baseline.py --seed 1

A snapshot of the training state is written every epoch and the run resumes from it when restarted.
"""
import argparse
import json
import os
import time

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_small

from augmentations import STORED_AUGMENTATIONS
from config import CLASS_NAMES, IMAGES_DIR, N_AUGMENTATIONS, NUM_CLASSES, ROOT, WEIGHTS_DIR, graph_path
from dataset import RESISCGraphs, init_worker
from test import plot_confusion_matrix

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # same input normalization as the VGG of the GNN
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class RESISCImages(Dataset):
    """Sample idx is the image of graph idx of RESISCGraphs (train: image idx // 51, augmentation idx % 51);
    val and test images are not augmented. Returns uint8 CHW tensors (4x less to move between processes than
    float), normalized on the GPU by to_input."""

    def __init__(self, split):
        self.split = split
        self.graphs = RESISCGraphs(split)  # rebuilds the images (data/images + seeded augmentation)
        self.labels = np.load(os.path.join(IMAGES_DIR, f"{split}_labels.npy"))

    def __len__(self):
        return len(self.graphs)

    def image(self, idx):
        if self.split == "train" and idx % N_AUGMENTATIONS in STORED_AUGMENTATIONS:
            return np.load(graph_path(self.split, idx))["img"]
        return self.graphs.image(idx)

    def __getitem__(self, idx):
        i = idx // N_AUGMENTATIONS if self.split == "train" else idx
        return torch.from_numpy(self.image(idx)).permute(2, 0, 1), int(self.labels[i])


def to_input(x, device):
    """uint8 batch -> normalized float batch on `device`."""
    return (x.to(device, non_blocking=True).float() / 255 - MEAN.to(device)) / STD.to(device)


def worker_init(worker_id):
    init_worker(worker_id)
    torch.set_num_threads(1)


CONFIG_KEYS = ("lr", "weight_decay", "label_smoothing")  # keys of the search configurations used here


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "trial_53.json"),
                   help="JSON of a SCH_HELM2 configuration whose lr, weight_decay and label_smoothing replace the "
                   "defaults ('' for none); arguments given explicitly still win")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=3e-6)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=30, help="early stopping patience (epochs)")
    p.add_argument("--plateau-patience", type=int, default=10, help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--eval-workers", type=int, default=2)
    p.add_argument("--experiment", default="RESISC45 - MobileNetV3-Small from scratch")
    args = p.parse_args()
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        p.set_defaults(**{k: config[k] for k in CONFIG_KEYS if k in config})
        args = p.parse_args()
    return args


def build_model():
    return mobilenet_v3_small(weights=None, num_classes=NUM_CLASSES)  # random initialization, no pretraining


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over `loader`; trains if an optimizer is given. Returns (mean loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = to_input(x, device), y.to(device, non_blocking=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (pred.argmax(dim=1) == y).sum().item()
            seen += len(y)
    return total_loss / seen, correct / seen


def test(model, loader, device, name):
    model.eval()
    preds, labels, batch_times = [], [], []
    wall_start = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            t0 = time.perf_counter()
            pred = model(to_input(x, device))  # the normalization is timed, as the VGG's in the GNN
            torch.cuda.synchronize()
            batch_times.append((time.perf_counter() - t0) / len(y))
            preds.append(pred.argmax(dim=1).cpu())
            labels.append(y)
    wall = time.perf_counter() - wall_start
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    print(f"Accuracy: {(preds == labels).mean():.4f}")
    for c, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name:<22} {(preds[labels == c] == c).mean():.4f}")
    cm_path = os.path.join(ROOT, f"{name}_confusion_matrix.png")
    plot_confusion_matrix(labels, preds, cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    per_image = np.array(batch_times) * 1e3
    print(f"\nInference: {len(preds)} images in {wall:.2f} s ({len(preds) / wall:.1f} images/s)")
    print(f"Forward pass per image: mean {per_image.mean():.3f} ms, std {per_image.std():.3f} ms, "
          f"min {per_image.min():.3f} ms, max {per_image.max():.3f} ms")
    return (preds == labels).mean()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    loaders = {split: DataLoader(RESISCImages(split), batch_size=args.batch_size, shuffle=split == "train",
                                 num_workers=args.workers if split == "train" else args.eval_workers,
                                 persistent_workers=True, pin_memory=True, worker_init_fn=worker_init)
               for split in ("train", "val", "test")}
    model = build_model().to(device)
    print(f"MobileNetV3-Small from scratch (no pretrained weights): {sum(p.numel() for p in model.parameters())} "
          f"parameters, all trainable")
    name = f"MobileNetV3_small_scratch_seed={args.seed}"
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(WEIGHTS_DIR, name + ".pth")
    snap_path = os.path.join(WEIGHTS_DIR, name + ".snap.pt")

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=args.plateau_patience,
                                                           min_lr=1e-6, factor=0.5)
    best_acc, best_epoch, no_improve, start_epoch, run_id = 0.0, -1, 0, 0, None
    if os.path.exists(snap_path):
        snap = torch.load(snap_path, map_location="cpu", weights_only=False)
        model.load_state_dict(snap["model"])
        optimizer.load_state_dict(snap["optimizer"])
        scheduler.load_state_dict(snap["scheduler"])
        best_acc, best_epoch, no_improve, run_id = snap["best_acc"], snap["best_epoch"], snap["no_improve"], snap["run_id"]
        start_epoch = snap["epoch"] + 1
        torch.set_rng_state(snap["cpu_rng"])
        torch.cuda.set_rng_state_all(snap["cuda_rng"])
        np.random.set_state(snap["np_rng"])
        print(f"Resuming from epoch {start_epoch + 1} ({snap_path})")

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=name, run_id=run_id) as run:
        if run_id is None:
            mlflow.log_params({**vars(args), "model": "mobilenet_v3_small", "weights": "none"})
        for epoch in range(start_epoch, args.epochs):
            if no_improve >= args.patience:  # stopped right before an interruption
                break
            m = {}
            m["train_loss"], m["train_acc"] = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
            m["val_loss"], m["val_acc"] = run_epoch(model, loaders["val"], loss_fn, device)
            m["test_loss"], m["test_acc"] = run_epoch(model, loaders["test"], loss_fn, device)
            m["lr"] = optimizer.param_groups[0]["lr"]
            mlflow.log_metrics(m, step=epoch)
            print(f"Epoch {epoch + 1:3d} | " + " | ".join(f"{k} {v:.4f}" for k, v in m.items()), flush=True)
            if m["val_acc"] > best_acc:
                best_acc, best_epoch = m["val_acc"], epoch
                torch.save(model.state_dict(), save_path)
                print(f"  new best val accuracy: {best_acc:.4f}, saved to {save_path}", flush=True)
            no_improve = 0 if best_epoch == epoch else no_improve + 1
            stop = no_improve >= args.patience
            if not stop:
                scheduler.step(m["val_acc"])
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                            best_acc=best_acc, best_epoch=best_epoch, no_improve=no_improve, epoch=epoch,
                            run_id=run.info.run_id, cpu_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all(), np_rng=np.random.get_state()), snap_path + ".tmp")
            os.replace(snap_path + ".tmp", snap_path)
            if stop:
                print(f"Early stopping after {epoch + 1} epochs (best val accuracy {best_acc:.4f} at epoch "
                      f"{best_epoch + 1})", flush=True)
                break

        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"\nTesting {name} (checkpoint of epoch {best_epoch + 1}, val accuracy {best_acc:.4f})")
        acc = test(model, loaders["test"], device, name)
        mlflow.log_metrics({"best_val_acc": best_acc, "best_epoch": best_epoch + 1, "final_test_acc": acc})
    os.remove(snap_path)


if __name__ == "__main__":
    main()
