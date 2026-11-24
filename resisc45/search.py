"""Hyperparameter search of SCH_HELM2 on RESISC45 with Optuna, on a subset of the training images (search_config.py)
and the validation set; the test set is not used. See run_search.sh for the whole sequence.

    python search.py check --fraction 0.1       # the graphs exist and the images rebuilt here match their maps
    python search.py run --trials 60            # TPE + Hyperband; the configuration in use is the first trial
    python search.py top --k 3                  # fixes the configurations to confirm and prints their superpixel counts
    python search.py confirm                    # those configurations and the one in use, with 5 new seeds

The study is in logs/search/study.db: running `run` again continues it, retrying the trial that was interrupted.
`confirm` skips the (configuration, seed) pairs already in logs/search/confirm.csv.

With --protocol (e.g. `python search.py --protocol tr20 run`), the search is done inside the labelled images of a
protocol of split_protocol.py (tr20: 140 per class), in logs/search_<protocol>/: the trials train on the training
images of the protocol and validate on its validation images (fold 0); the confirmation run with seed s validates on
fold s - 1 (folds 1-5, images the trials never validated on) and trains on the other labelled images. Besides the
configuration in use, configs/trial_53.json (the best one of the search above) is a trial of the study and is confirmed
as ref_trial_53. The test images of the protocol are not used.
"""
import argparse
import csv
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
from dataset import RESISCGraphs, init_worker
from search_config import (CONFIRM_FRACTION, NODE_VALUES, SEARCH_FRACTION, graph_file, protocol_labelled_graphs,
                           protocol_train_graphs, protocol_val_graphs, train_graphs, val_graphs)
from train import build_model, run_epoch

CONFIRM_SEEDS = (2, 3, 4, 5, 6)  # not used in the search (seed 1) nor in the full runs

# The configuration in use: train.py's defaults with the readout of the full runs (lse), 100 superpixels
BASELINE = dict(n_nodes=100, percentile=50, threshold="batch", lr=1e-3, dropout=0.1, heads=2, embedding_size=64,
                num_clusters=16, label_smoothing=0.1, weight_decay=3e-6)
# Configurations also tried first and confirmed in the search of a protocol: the best one of the search on the
# torchgeo splits
PROTOCOL_REFERENCES = {"ref_trial_53": os.path.join(ROOT, "configs", "trial_53.json")}


def study_setup(protocol):
    """Where the study of the torchgeo splits (protocol None) or of a protocol keeps its results, and its names."""
    tag = f" - {protocol}" if protocol else ""
    results = os.path.join(ROOT, "logs", f"search_{protocol}" if protocol else "search")
    references = {}
    if protocol:
        for name, path in PROTOCOL_REFERENCES.items():
            with open(path) as f:
                references[name] = json.load(f)
    return Namespace(protocol=protocol, results=results, storage=f"sqlite:///{os.path.join(results, 'study.db')}",
                     study="SCH_HELM2 RESISC45" + tag, experiment="RESISC45 - SCH_HELM2 - Optuna search" + tag,
                     references=references)


def run_data(setup, seed, confirm):
    """(training graphs, validation split, validation graphs, parameters to log) of a trial (confirm=False, seed 1)
    or of a confirmation run, all in the train split except the validation set of the torchgeo splits."""
    if setup.protocol is None:
        fraction = CONFIRM_FRACTION if confirm else SEARCH_FRACTION
        return train_graphs(fraction), "val", val_graphs(), {"fraction": fraction}
    fold = seed - 1 if confirm else 0
    return (protocol_train_graphs(setup.protocol, fold), "train", protocol_val_graphs(setup.protocol, fold),
            {"protocol": setup.protocol, "fold": fold})


def suggest(trial):
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


class SearchGraphs(RESISCGraphs):
    """Some graphs of a split, with a given number of superpixels."""

    def __init__(self, split, n_nodes, graph_ids, verify=False):
        super().__init__(split, verify)
        self.n_nodes = n_nodes
        self.graph_ids = graph_ids

    def __len__(self):
        return len(self.graph_ids)

    def path(self, idx):
        return graph_file(self.split, idx, self.n_nodes)

    def __getitem__(self, idx):
        return super().__getitem__(self.graph_ids[idx])


def train_config(params, seed, data, epochs, plateau_patience, patience, workers, run_name, trial=None):
    """Trains one configuration on `data` (run_data) and returns its best validation accuracy. With a trial, it
    reports the validation accuracy of each epoch and may be pruned."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda")
    n = params["n_nodes"]
    train_ids, val_split, val_ids, data_params = data
    train_loader = DataLoader(SearchGraphs("train", n, train_ids), batch_size=64, shuffle=True,
                              num_workers=workers, persistent_workers=True, worker_init_fn=init_worker)
    val_loader = DataLoader(SearchGraphs(val_split, n, val_ids), batch_size=64, num_workers=2,
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
        mlflow.log_params({**params, "seed": seed, **data_params})
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


def load_study(setup, sampler=None):
    os.makedirs(setup.results, exist_ok=True)
    # A trial without heartbeat for 3 min (its process died) is marked failed and tried again
    storage = optuna.storages.RDBStorage(setup.storage, heartbeat_interval=60, grace_period=180,
                                         failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=2))
    return optuna.create_study(study_name=setup.study, storage=storage, direction="maximize", load_if_exists=True,
                               sampler=sampler, pruner=optuna.pruners.HyperbandPruner(min_resource=5, max_resource=30,
                                                                                      reduction_factor=3))


def export_trials(setup, study):
    study.trials_dataframe(attrs=("number", "value", "state", "params", "duration")).to_csv(
        os.path.join(setup.results, "trials.csv"), index=False)


def cmd_check(setup, args):
    if setup.protocol:
        sets = (("train", protocol_labelled_graphs(setup.protocol)),)  # every fold
    elif args.fraction is None:
        raise SystemExit("check: --fraction is needed without --protocol")
    else:
        sets = (("val", val_graphs()), ("train", train_graphs(args.fraction)))
    for n in args.nodes:
        for split, ids in sets:
            missing = sum(not os.path.exists(graph_file(split, i, n)) for i in ids)
            if missing:
                raise SystemExit(f"{split}, {n} superpixels: {missing} of {len(ids)} graphs missing")
            ds = SearchGraphs(split, n, random.Random(0).sample(ids, 60), verify=True)
            sizes = [ds[i].num_nodes for i in range(len(ds))]  # verify=True raises if an image differs from its map
            print(f"{split}, {n} superpixels: {len(ids)} graphs, 60 checked, nodes {min(sizes)}-{max(sizes)}", flush=True)
    print("OK")


def cmd_run(setup, args):
    # Different sampler seed per process and after a restart, so the random first trials are not repeated
    done = len(optuna.load_study(study_name=setup.study, storage=setup.storage).trials) \
        if os.path.exists(setup.storage[10:]) else 0
    study = load_study(setup, optuna.samplers.TPESampler(seed=args.worker + 1000 * done + 42, multivariate=True,
                                                         group=True, constant_liar=True, n_startup_trials=15))
    for params in (BASELINE, *setup.references.values()):
        study.enqueue_trial(dict(params), skip_if_exists=True)
    mlflow.set_experiment(setup.experiment)
    data = run_data(setup, seed=1, confirm=False)

    def objective(trial):
        return train_config(suggest(trial), seed=1, data=data, epochs=30, plateau_patience=5, patience=10,
                            workers=args.workers, run_name=f"trial_{trial.number}", trial=trial)

    states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    study.optimize(objective, callbacks=[optuna.study.MaxTrialsCallback(args.trials, states=states),
                                         lambda s, _: export_trials(setup, s)],
                   catch=(torch.cuda.OutOfMemoryError,), gc_after_trial=True)
    export_trials(setup, study)
    print(f"search finished: {len(study.trials)} trials, best val_acc {study.best_value:.4f} in trial "
          f"{study.best_trial.number}: {study.best_params}")


def confirm_path(setup):
    return os.path.join(setup.results, "confirm_configs.json")


def cmd_top(setup, args):
    """Fixes the configurations to confirm (once): the one in use, the references and the k best other trials;
    prints their superpixel counts."""
    if not os.path.exists(confirm_path(setup)):
        study = load_study(setup)
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        complete.sort(key=lambda t: t.value, reverse=True)
        configs = {"baseline": BASELINE, **setup.references}
        fixed = list(configs.values())
        for t in complete:
            if len(configs) - len(fixed) >= args.k:
                break
            if t.params not in fixed:
                configs[f"trial_{t.number}"] = t.params
        with open(confirm_path(setup), "w") as f:
            json.dump(configs, f, indent=2)
    with open(confirm_path(setup)) as f:
        print(*sorted({c["n_nodes"] for c in json.load(f).values()}))


def cmd_confirm(setup, args):
    with open(confirm_path(setup)) as f:
        configs = json.load(f)
    path = os.path.join(setup.results, "confirm.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {(r["config"], int(r["seed"])) for r in csv.DictReader(f)}
    mlflow.set_experiment(setup.experiment + " - confirmation")
    for seed in CONFIRM_SEEDS:
        data = run_data(setup, seed, confirm=True)
        for name, params in configs.items():
            if (name, seed) in done:
                continue
            acc = train_config(params, seed, data, epochs=30, plateau_patience=5, patience=10,
                               workers=args.workers, run_name=f"{name}_seed={seed}")
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["config", "seed", "best_val_acc"])
                w.writerow([name, seed, acc])
            print(f"finished {name} seed {seed}: val_acc {acc:.4f}", flush=True)
    results = pd.read_csv(path).pivot(index="seed", columns="config", values="best_val_acc") * 100
    print(results.round(2).to_string())
    if setup.protocol:
        print(f"(seed s: validation on fold s - 1 of {setup.protocol}, {len(run_data(setup, 2, True)[2])} images)")
    for ref in ("baseline", *setup.references):
        print(f"\nmean ± std and paired difference to {ref} (pp):")
        for name in results.columns:
            diff = results[name] - results[ref]
            print(f"  {name:12s} {results[name].mean():.2f} ± {results[name].std():.2f} | "
                  f"diff {diff.mean():+.2f} ± {diff.std():.2f} | better in {(diff > 0).sum()}/{len(diff)} seeds")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", help="search inside the labelled images of splits/<protocol>.npz (split_protocol.py)")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("--fraction", type=float, help="subset of the training images (without --protocol)")
    c.add_argument("--nodes", type=int, nargs="*", default=list(NODE_VALUES))
    c = sub.add_parser("run")
    c.add_argument("--trials", type=int, default=60)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--worker", type=int, default=0, help="number of this process, when several run the same study")
    c = sub.add_parser("top")
    c.add_argument("--k", type=int, default=3)
    c = sub.add_parser("confirm")
    c.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    setup = study_setup(args.protocol)
    {"check": cmd_check, "run": cmd_run, "top": cmd_top, "confirm": cmd_confirm}[args.command](setup, args)


if __name__ == "__main__":
    main()
