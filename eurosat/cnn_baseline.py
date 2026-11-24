"""CNN baseline: MobileNetV3-Small (torchvision, 1.0x) trained from scratch on EuroSAT, with the protocol of the
full SCH_HELM2 runs: the same splits, the same 51 training augmentations per image (rebuilt with the seeds of
build_dataset.py), all of them in every epoch, Adam (lr 1e-3, betas 0.9/0.98, weight decay 3e-6), label smoothing
0.1, batch 64, ReduceLROnPlateau patience 10, early stopping patience 30 on the validation accuracy. No pretrained
weights. After training, the checkpoint with the best validation accuracy is evaluated on the test set.

    python cnn_baseline.py --seed 1

A snapshot of the training state is written every epoch and the run resumes from it when restarted.
"""
import argparse
import contextlib
import os
import random
import time

import mlflow
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_small

from augmentations import AUGMENTATIONS
from config import CLASS_NAMES, INDICES_DIR, N_AUGMENTATIONS, NUM_CLASSES, RAW_DIR, ROOT, WEIGHTS_DIR
from test import plot_confusion_matrix

# torchvision EuroSAT (eurosat/2750/<class>/*.jpg) the graphs were built from (build_dataset.py, data/raw)
IMAGES_DIR = os.path.join(RAW_DIR, "eurosat", "2750")
IMAGES = os.path.join(ROOT, "data", "eurosat_images.npy")  # all 27000 images decoded once, torchvision order
LABELS = os.path.join(ROOT, "data", "eurosat_labels.npy")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # same input normalization as the VGG of the GNN
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def decode_images():
    """Decodes the dataset once, in the order of torchvision's EuroSAT (classes and files sorted), the one the
    indices/*.npy refer to."""
    if os.path.exists(IMAGES):
        return
    classes = sorted(os.listdir(IMAGES_DIR))
    files = [(os.path.join(IMAGES_DIR, c, f), y) for y, c in enumerate(classes)
             for f in sorted(os.listdir(os.path.join(IMAGES_DIR, c)))]
    images = np.stack([np.asarray(Image.open(path).convert("RGB")) for path, _ in files])
    os.makedirs(os.path.dirname(IMAGES), exist_ok=True)
    np.save(LABELS, np.array([y for _, y in files], dtype=np.int64))
    np.save(IMAGES + ".tmp.npy", images)
    os.replace(IMAGES + ".tmp.npy", IMAGES)


@contextlib.contextmanager
def seeded(seed):
    """The python/numpy/torch seeds of build_dataset.py inside the block; the previous RNG states are restored."""
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


class EuroSATImages(Dataset):
    """Training sample k of image i is augmentation k with seed i * N_AUGMENTATIONS + k, as graph
    train_{i * N_AUGMENTATIONS + k} of build_dataset.py; val and test images are not augmented."""

    def __init__(self, split):
        self.split = split
        self.indices = np.load(os.path.join(INDICES_DIR, f"{split}_indices.npy"))
        self.labels = np.load(LABELS)
        self.size = len(self.indices) * (N_AUGMENTATIONS if split == "train" else 1)
        self._images = None  # opened lazily, once per DataLoader worker

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self._images is None:
            self._images = np.load(IMAGES, mmap_mode="r")
        i, k = divmod(idx, N_AUGMENTATIONS) if self.split == "train" else (idx, None)
        j = int(self.indices[i])
        image = np.array(self._images[j])
        if k is not None:
            with seeded(idx):
                image = np.ascontiguousarray(AUGMENTATIONS[k][0](image), dtype=np.uint8)
        x = (torch.from_numpy(image).permute(2, 0, 1).float() / 255 - MEAN) / STD
        return x, int(self.labels[j])


def init_worker(_):
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=30, help="early stopping patience (epochs)")
    p.add_argument("--plateau-patience", type=int, default=10, help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval-workers", type=int, default=2)
    p.add_argument("--experiment", default="EuroSAT - MobileNetV3-Small from scratch")
    return p.parse_args()


def build_model():
    return mobilenet_v3_small(weights=None, num_classes=NUM_CLASSES)  # random initialization, no pretraining


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over `loader`; trains if an optimizer is given. Returns (mean loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
            pred = model(x)
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
    decode_images()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    loaders = {split: DataLoader(EuroSATImages(split), batch_size=args.batch_size, shuffle=split == "train",
                                 num_workers=args.workers if split == "train" else args.eval_workers,
                                 persistent_workers=True, pin_memory=True, worker_init_fn=init_worker)
               for split in ("train", "val", "test")}
    model = build_model().to(device)
    print(f"MobileNetV3-Small from scratch (no pretrained weights): {sum(p.numel() for p in model.parameters())} "
          f"parameters, all trainable")
    name = f"MobileNetV3_small_scratch_seed={args.seed}"
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(WEIGHTS_DIR, name + ".pth")
    snap_path = os.path.join(WEIGHTS_DIR, name + ".snap.pt")

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=3e-6)
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
