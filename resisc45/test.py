"""Evaluates a trained model on the test split: accuracy, per-class accuracy, confusion
matrix and inference time. Takes the same arguments as train.py to rebuild the model.

    python test.py --seed 1
    python test.py --seed 1 --readout lse --config configs/trial_53.json
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader

from config import CLASS_NAMES, ROOT, WEIGHTS_DIR
from dataset import graphs, init_worker
from train import build_model, count_parameters, parse_args, run_name


def plot_confusion_matrix(y_true, y_pred, path):
    """Row-normalized confusion matrix, colored on a log scale from 1e-3 to 1."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    cm = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(30, 26))
    ax = sns.heatmap(np.log10(cm + 1e-5), annot=cm, fmt=".2f", cmap="Blues", vmin=-3, vmax=0,
                     xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar_kws={"shrink": 0.65, "aspect": 30})
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-3, -2, -1, 0])
    colorbar.set_ticklabels(["10$^{-3}$", "10$^{-2}$", "10$^{-1}$", "10$^{0}$"])
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(graphs("test", args.nodes, args.protocol), batch_size=args.batch_size, num_workers=args.workers,
                        worker_init_fn=init_worker)

    model = build_model(args)
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, run_name(args) + ".pth"), map_location="cpu"))
    model = model.to(device).eval()
    print(f"Testing {run_name(args)} ({count_parameters(model)} trainable parameters)")

    preds, labels, batch_times = [], [], []
    wall_start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            t0 = time.perf_counter()
            pred = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_times.append((time.perf_counter() - t0) / batch.num_graphs)
            preds.append(pred.argmax(dim=1).cpu())
            labels.append(batch.y.cpu())
    wall = time.perf_counter() - wall_start
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()

    print(f"Accuracy: {(preds == labels).mean():.4f}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:<22} {(preds[labels == c] == c).mean():.4f}")

    cm_path = os.path.join(ROOT, f"{run_name(args)}_confusion_matrix.png")
    plot_confusion_matrix(labels, preds, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    per_image = np.array(batch_times) * 1e3
    print(f"\nInference: {len(preds)} images in {wall:.2f} s ({len(preds) / wall:.1f} images/s)")
    print(f"Forward pass per image: mean {per_image.mean():.3f} ms, std {per_image.std():.3f} ms, "
          f"min {per_image.min():.3f} ms, max {per_image.max():.3f} ms")


if __name__ == "__main__":
    main()
