"""Hyperparameter search of SCH_HELM2 (SLC-HELM) on EuroSAT with Optuna, on a subset of the training images
(search_config.py) and the validation set; the test set is not used. See run_search.sh for the whole sequence.

    python search.py check --fraction 0.1       # the graphs of the search exist and load
    python search.py run --trials 60            # TPE + Hyperband; the reference configuration is the first trial
    python search.py top --k 3                  # fixes the configurations to confirm and prints their superpixel counts
    python search.py confirm                    # those configurations and the reference, with 5 new seeds
    python search.py export trial_12            # writes a confirmed configuration to configs/trial_12.json

Same protocol as the RESISC45 and AID searches: 60 trials of up to 30 epochs on 10% of the training images (8
augmentations each), seed 1, multivariate TPE sampler and Hyperband pruner, validation accuracy maximized; then the 3
best configurations and the reference are trained again with 5 new seeds on 30% of the training images, and the one
with the best mean validation accuracy is kept. Picking the best of many single-seed trials inflates its accuracy, so
only the confirmation compares them.

The study is in logs/search/study.db: running `run` again continues it, retrying the trial that was interrupted.
`confirm` skips the (configuration, seed) pairs already in logs/search/confirm.csv. A trial that runs out of GPU
memory is pruned (it tells the sampler the region is bad) without counting as a trial.
"""
import argparse
import csv
import gc
import json
import os
import random
from argparse import Namespace

import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from config import ROOT
from dataset import EuroSATGraphs, init_worker
from search_config import CONFIRM_FRACTION, NODE_VALUES, SEARCH_FRACTION, graph_file, train_graphs, val_graphs
from train import build_model, run_epoch

CONFIRM_SEEDS = (2, 3, 4, 5, 6)  # not used in the search (seed 1) nor in the full runs (1 16 20 58 123)
RESULTS = os.path.join(ROOT, "logs", "search")
STORAGE = f"sqlite:///{os.path.join(RESULTS, 'study.db')}"
STUDY = "SCH_HELM2 EuroSAT"
EXPERIMENT = "EuroSAT - SCH_HELM2 - Optuna search"
# Reference configuration: the one of the paper's runs (train.py's defaults, 50 superpixels). It is the first trial and
# is confirmed together with the best trials.
REFERENCES = {"default": os.path.join(ROOT, "configs", "default.json")}


def suggest(trial):
    """Search space (same as RESISC45; AID searches embedding sizes 32 and 64 only, 128 does not fit in 24 GB)."""
    return dict(
        n_nodes=trial.suggest_categorical("n_nodes", NODE_VALUES),
        percentile=trial.suggest_float("percentile", 20, 80, step=5),
        threshold=trial.suggest_categorical("threshold", ("batch", "graph")),
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        dropout=trial.suggest_float("dropout", 0, 0.4, step=0.05),
        heads=trial.suggest_categorical("heads", (1, 2, 4)),
        embedding_size=trial.suggest_categorical("embedding_size", (32, 64, 128)),
        num_clusters=trial.suggest_categorical("num_clusters", (8, 16, 32)),
        label_smoothing=trial.suggest_float("label_smoothing", 0, 0.2, step=0.05),
        weight_decay=trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
    )


def references():
    configs = {}
    for name, path in REFERENCES.items():
        with open(path) as f:
            configs[name] = json.load(f)
    return configs


class SearchGraphs(EuroSATGraphs):
    """Some graphs of a split, with a given number of superpixels."""

    def __init__(self, split, n_nodes, graph_ids):
        super().__init__(split, n_nodes)
        self.graph_ids = graph_ids

    def __len__(self):
        return len(self.graph_ids)

    def path(self, idx):
        return graph_file(self.split, idx, self.n_nodes)

    def __getitem__(self, idx):
        return super().__getitem__(self.graph_ids[idx])


def train_config(params, seed, train_ids, epochs, plateau_patience, patience, workers, run_name, batch_size,
                 trial=None):
    """Trains one configuration on the graphs train_ids (train split) and returns its best validation accuracy. With
    a trial, it reports the validation accuracy of each epoch and may be pruned."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda")
    n = params["n_nodes"]
    train_loader = DataLoader(SearchGraphs("train", n, train_ids), batch_size=batch_size, shuffle=True,
                              num_workers=workers, persistent_workers=workers > 0, worker_init_fn=init_worker)
    val_loader = DataLoader(SearchGraphs("val", n, val_graphs()), batch_size=batch_size, num_workers=2,
                            persistent_workers=True, worker_init_fn=init_worker)
    model = build_model(Namespace(aggregation="netvlad", readout="lse", mgp="branches", **{
        k: params[k] for k in ("embedding_size", "dropout", "heads", "percentile", "num_clusters", "threshold")})).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=params["label_smoothing"])
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=params["lr"],
                                 betas=(0.9, 0.98), eps=1e-9, weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=plateau_patience,
                                                           min_lr=1e-6, factor=0.5)
    best, no_improve, pruned = 0.0, 0, False
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**params, "seed": seed, "train_graphs": len(train_ids)})
        for epoch in range(epochs):
            train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
            mlflow.log_metrics(dict(train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc,
                                    lr=optimizer.param_groups[0]["lr"]), step=epoch)
            print(f"{run_name} epoch {epoch + 1:2d} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                  f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}", flush=True)
            if not np.isfinite(train_loss):  # diverged: this configuration will not recover
                pruned = trial is not None
                break
            no_improve = 0 if val_acc > best else no_improve + 1
            best = max(best, val_acc)
            if trial is not None:
                trial.report(val_acc, epoch)
                if trial.should_prune():
                    pruned = True
                    break
            if no_improve >= patience:
                break
            scheduler.step(val_acc)
        mlflow.log_metric("best_val_acc", best)
        mlflow.set_tag("pruned", pruned)
    del train_loader, val_loader, model, optimizer
    torch.cuda.empty_cache()
    if pruned:
        raise optuna.TrialPruned()
    return best


def load_study(sampler=None):
    os.makedirs(RESULTS, exist_ok=True)
    # A trial without heartbeat for 3 min (its process died) is marked failed and tried again
    storage = optuna.storages.RDBStorage(
        STORAGE, heartbeat_interval=60, grace_period=180,
        heartbeat_stale_trial_callback=optuna.storages.RetryHeartbeatStaleTrialCallback(max_retry=2))
    return optuna.create_study(study_name=STUDY, storage=storage, direction="maximize", load_if_exists=True,
                               sampler=sampler, pruner=optuna.pruners.HyperbandPruner(min_resource=5, max_resource=30,
                                                                                      reduction_factor=3))


def export_trials(study):
    study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration")).to_csv(
        os.path.join(RESULTS, "trials.csv"), index=False)


def finished_trials(study):
    """Trials of the budget: complete, or pruned by the pruner (not by running out of GPU memory)."""
    states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    return sum(not t.user_attrs.get("oom") for t in study.get_trials(deepcopy=False, states=states))


def cmd_check(args):
    sets = (("val", val_graphs()), ("train", train_graphs(args.fraction)))
    for n in args.nodes or NODE_VALUES:
        for split, ids in sets:
            missing = sum(not os.path.exists(graph_file(split, i, n)) for i in ids)
            if missing:
                raise SystemExit(f"{split}, {n} superpixels: {missing} of {len(ids)} graphs missing (search_build.py)")
            ds = SearchGraphs(split, n, random.Random(0).sample(ids, 60))
            sizes = [ds[i].num_nodes for i in range(len(ds))]
            print(f"{split}, {n} superpixels: {len(ids)} graphs, 60 loaded, nodes {min(sizes)}-{max(sizes)}", flush=True)
    print("OK")


def cmd_run(args):
    # Different sampler seed per process and after a restart, so the random first trials are not repeated
    done = len(optuna.load_study(study_name=STUDY, storage=STORAGE).trials) if os.path.exists(STORAGE[10:]) else 0
    study = load_study(optuna.samplers.TPESampler(seed=args.worker + 1000 * done + 42, multivariate=True, group=True,
                                                  constant_liar=True, n_startup_trials=15))
    for params in references().values():
        study.enqueue_trial(dict(params), skip_if_exists=True)
    mlflow.set_experiment(EXPERIMENT)
    train_ids = train_graphs(SEARCH_FRACTION)

    def objective(trial):
        try:
            return train_config(suggest(trial), seed=1, train_ids=train_ids, epochs=30, plateau_patience=5,
                                patience=10, workers=args.workers, run_name=f"trial_{trial.number}",
                                batch_size=args.batch_size, trial=trial)
        except torch.cuda.OutOfMemoryError:
            pass
        # Outside the except block, so the frames of train_config (model, loaders) can be freed
        gc.collect()
        torch.cuda.empty_cache()
        trial.set_user_attr("oom", True)
        print(f"trial_{trial.number}: out of GPU memory, pruned", flush=True)
        raise optuna.TrialPruned("out of GPU memory")

    def stop_when_done(s, _):
        export_trials(s)
        if finished_trials(s) >= args.trials:
            s.stop()

    if finished_trials(study) < args.trials:  # the callback only runs after a trial: a finished study runs none
        study.optimize(objective, callbacks=[stop_when_done], catch=(torch.cuda.OutOfMemoryError,), gc_after_trial=True)
    export_trials(study)
    print(f"search finished: {len(study.trials)} trials, best val_acc {study.best_value:.4f} in trial "
          f"{study.best_trial.number}: {study.best_params}")


def confirm_path():
    return os.path.join(RESULTS, "confirm_configs.json")


def cmd_top(args):
    """Fixes the configurations to confirm (once): the references and the k best other trials; prints their
    superpixel counts."""
    if not os.path.exists(confirm_path()):
        study = load_study()
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        complete.sort(key=lambda t: t.value, reverse=True)
        configs = references()
        fixed = list(configs.values())
        for t in complete:
            if len(configs) - len(fixed) >= args.k:
                break
            if t.params not in fixed:
                configs[f"trial_{t.number}"] = t.params
        with open(confirm_path(), "w") as f:
            json.dump(configs, f, indent=2)
    with open(confirm_path()) as f:
        print(*sorted({c["n_nodes"] for c in json.load(f).values()}))


def cmd_confirm(args):
    with open(confirm_path()) as f:
        configs = json.load(f)
    path = os.path.join(RESULTS, "confirm.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {(r["config"], int(r["seed"])) for r in csv.DictReader(f)}
    mlflow.set_experiment(EXPERIMENT + " - confirmation")
    train_ids = train_graphs(CONFIRM_FRACTION)
    for seed in CONFIRM_SEEDS:
        for name, params in configs.items():
            if (name, seed) in done:
                continue
            acc = train_config(params, seed, train_ids, epochs=30, plateau_patience=5, patience=10,
                               workers=args.workers, run_name=f"{name}_seed={seed}", batch_size=args.batch_size)
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["config", "seed", "best_val_acc"])
                w.writerow([name, seed, acc])
            print(f"finished {name} seed {seed}: val_acc {acc:.4f}", flush=True)
    results = pd.read_csv(path).pivot(index="seed", columns="config", values="best_val_acc") * 100
    print(results.round(2).to_string())
    for ref in REFERENCES:
        print(f"\nmean ± std and paired difference to {ref} (pp):")
        for name in results.columns:
            diff = results[name] - results[ref]
            print(f"  {name:16s} {results[name].mean():.2f} ± {results[name].std():.2f} | "
                  f"diff {diff.mean():+.2f} ± {diff.std():.2f} | better in {(diff > 0).sum()}/{len(diff)} seeds")


def cmd_export(args):
    """Writes a configuration of the confirmation to configs/<name>.json (train.py --config, CONFIG= of the scripts)."""
    with open(confirm_path()) as f:
        configs = json.load(f)
    if args.name not in configs:
        raise SystemExit(f"{args.name} is not among the confirmed configurations: {sorted(configs)}")
    out = os.path.join(ROOT, "configs", f"{args.name}.json")
    if os.path.exists(out) and not args.force:
        raise SystemExit(f"{out} exists (--force to overwrite)")
    with open(out, "w") as f:
        json.dump(configs[args.name], f, indent=2)
        f.write("\n")
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("--fraction", type=float, required=True, help="subset of the training images")
    c.add_argument("--nodes", type=int, nargs="*", help="default: the superpixel counts of the search")
    c = sub.add_parser("run")
    c.add_argument("--trials", type=int, default=60)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--batch-size", type=int, default=64)
    c.add_argument("--worker", type=int, default=0, help="number of this process, when several run the same study")
    c = sub.add_parser("top")
    c.add_argument("--k", type=int, default=3)
    c = sub.add_parser("confirm")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--batch-size", type=int, default=64)
    c = sub.add_parser("export")
    c.add_argument("name", help="configuration of logs/search/confirm_configs.json, e.g. trial_12")
    c.add_argument("--force", action="store_true")
    args = p.parse_args()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    {"check": cmd_check, "run": cmd_run, "top": cmd_top, "confirm": cmd_confirm, "export": cmd_export}[args.command](args)


if __name__ == "__main__":
    main()
