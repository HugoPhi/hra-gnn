from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import load_dataset, load_provided_splits, make_splits
from .metrics import anomaly_metrics, normal_score_threshold
from .utils import parameter_count, resolve_device, seed_everything


def _official_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official model from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _splits(config: dict[str, Any], dataset) -> dict[str, list[int]]:
    split = config["dataset"].get("split", {})
    if split.get("mode", "generated") == "provided":
        return load_provided_splits(
            dataset,
            config["dataset"]["root"],
            train_file=split.get("train_file", "train_ids.txt"),
            validation_file=split.get("validation_file"),
            test_file=split.get("test_file", "test_ids.txt"),
        )
    return make_splits(
        dataset,
        train_normal_ratio=split.get("train_normal_ratio", 0.6),
        validation_normal_ratio=split.get("validation_normal_ratio", 0.2),
        validation_anomaly_ratio=split.get("validation_anomaly_ratio", 0.0),
        seed=int(config["training"].get("seed", 42)),
    )


def _to_pyg_graphs(
    dataset, indices: list[int], *, rw_dim: int, dg_dim: int, mamba: bool
):
    try:
        from scipy import sparse as sp
        from scipy.sparse import csgraph
        from torch_geometric.data import Data
        from torch_geometric.utils import degree, to_scipy_sparse_matrix
    except ImportError as exc:
        raise RuntimeError(
            "Official baselines require requirements-baselines.txt"
        ) from exc

    graphs = {}
    for index in indices:
        source = dataset[index]
        graph = Data(
            x=source.x.float(),
            edge_index=source.edge_index.long(),
            edge_type=source.edge_type.long(),
            y=torch.tensor([source.label], dtype=torch.long),
            graph_id=torch.tensor([source.graph_id], dtype=torch.long),
        )
        adjacency = to_scipy_sparse_matrix(
            graph.edge_index, num_nodes=graph.num_nodes
        )
        degrees = degree(graph.edge_index[0], num_nodes=graph.num_nodes)
        inverse = degrees.clamp_min(1).reciprocal().numpy()
        transition = adjacency * sp.diags(inverse)
        power = transition
        random_walk = [torch.from_numpy(power.diagonal()).float()]
        for _ in range(rw_dim - 1):
            power = power * transition
            random_walk.append(torch.from_numpy(power.diagonal()).float())
        random_walk_encoding = torch.stack(random_walk, dim=-1)
        clipped_degrees = degrees.numpy().clip(0, dg_dim - 1).astype(int)
        degree_encoding = torch.zeros((graph.num_nodes, dg_dim))
        degree_encoding[torch.arange(graph.num_nodes), clipped_degrees] = 1
        graph.x_s = torch.cat([random_walk_encoding, degree_encoding], dim=1)
        if mamba:
            laplacian = torch.from_numpy(
                csgraph.laplacian(adjacency, normed=True).toarray()
            ).float()
            graph.XLX_f = torch.diag(graph.x.T @ laplacian @ graph.x).unsqueeze(0)
            graph.XLX_s = torch.diag(
                graph.x_s.T @ laplacian @ graph.x_s
            ).unsqueeze(0)
        graphs[index] = graph
    return graphs


def _limited_splits(dataset, splits, maximum):
    if maximum is None:
        return splits
    limited = {}
    for split, indices in splits.items():
        by_label: dict[int, list[int]] = {}
        for index in indices:
            by_label.setdefault(int(dataset[index].label), []).append(index)
        if len(by_label) == 1:
            limited[split] = indices[:maximum]
            continue
        per_label = max(1, maximum // len(by_label))
        selected = []
        for label in sorted(by_label):
            selected.extend(by_label[label][:per_label])
        limited[split] = selected[:maximum]
    return limited


def _evaluation_loader(graphs, batch_size):
    from torch_geometric.loader import DataLoader

    batches = [
        list(range(start, min(start + batch_size, len(graphs))))
        for start in range(0, len(graphs), batch_size)
    ]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-1].insert(0, batches[-2].pop())
    return DataLoader(graphs, batch_sampler=batches)


def _score_loader(model, loader, device, args, mamba, statistics):
    labels: list[int] = []
    scores: list[float] = []
    graph_ids: list[int] = []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if mamba:
                outputs = model(
                    data,
                    data.x,
                    data.x_s,
                    data.edge_index,
                    data.batch,
                    data.num_graphs,
                    args,
                )
            else:
                outputs = model(
                    data.x,
                    data.x_s,
                    data.edge_index,
                    data.batch,
                    data.num_graphs,
                )
            graph_loss = model.calc_loss_g(outputs[0], outputs[1])
            node_loss = model.calc_loss_n(outputs[2], outputs[3], data.batch)
            if args.is_adaptive:
                score = (
                    (graph_loss - statistics["mean_g"]) / statistics["std_g"]
                    + (node_loss - statistics["mean_n"]) / statistics["std_n"]
                )
            else:
                score = graph_loss + node_loss
            labels.extend(data.y.reshape(-1).cpu().tolist())
            scores.extend(score.detach().cpu().tolist())
            graph_ids.extend(data.graph_id.reshape(-1).cpu().tolist())
    return labels, scores, graph_ids


def run_dual_view_fair(
    config: dict[str, Any],
    *,
    architecture: str,
    external_root: str | Path = "external",
) -> dict[str, Any]:
    if architecture not in {"cvtgad", "gladmamba"}:
        raise ValueError("architecture must be cvtgad or gladmamba")
    try:
        from torch_geometric.loader import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "Official baselines require requirements-baselines.txt"
        ) from exc

    baseline = config.get("recent_baseline", {})
    seed = int(config["training"].get("seed", 42))
    seed_everything(seed)
    device = resolve_device(config["training"].get("device", "auto"))
    dataset = load_dataset(config["dataset"])
    dataset_splits = _splits(config, dataset)
    dataset_splits = _limited_splits(
        dataset,
        dataset_splits,
        baseline.get("max_graphs_per_split"),
    )
    mamba = architecture == "gladmamba"
    rw_dim = int(baseline.get("rw_dim", 16))
    dg_dim = int(baseline.get("dg_dim", 16))
    selected_indices = sorted(
        {index for indices in dataset_splits.values() for index in indices}
    )
    graphs = _to_pyg_graphs(
        dataset,
        selected_indices,
        rw_dim=rw_dim,
        dg_dim=dg_dim,
        mamba=mamba,
    )
    batch_size = int(config["training"].get("batch_size", 32))
    if batch_size < 2:
        raise ValueError(
            "CVTGAD/GLADMamba need batch_size >= 2 for graph contrastive negatives"
        )
    train_loader = DataLoader(
        [graphs[index] for index in dataset_splits["train"]],
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    train_eval_loader = _evaluation_loader(
        [graphs[index] for index in dataset_splits["train"]], batch_size
    )
    test_loader = _evaluation_loader(
        [graphs[index] for index in dataset_splits["test"]],
        int(config["evaluation"].get("batch_size", batch_size)),
    )

    official_name = "GLADMamba" if mamba else "CVTGAD"
    module = _official_module(
        Path(external_root) / official_name / "model.py",
        f"official_{architecture}_{seed}",
    )
    arguments = SimpleNamespace(
        GNN_Encoder=baseline.get("gnn_encoder", "GCN" if mamba else "GIN"),
        graph_level_pool=baseline.get("graph_level_pool", "global_mean_pool"),
        is_adaptive=int(baseline.get("adaptive", 1)),
        alpha=float(baseline.get("alpha", 0.0)),
        d_model=int(baseline.get("d_model", 64)),
        dt_rank=int(baseline.get("dt_rank", 4)),
        d_state=int(baseline.get("d_state", 4)),
        d_conv=int(baseline.get("d_conv", 4)),
        conv_bias=bool(baseline.get("conv_bias", True)),
        bias=bool(baseline.get("bias", True)),
        l=int(baseline.get("l", 5)),
    )
    model_class = getattr(module, official_name)
    model = model_class(
        int(baseline.get("hidden_dim", 16)),
        int(baseline.get("num_layers", 5)),
        int(next(iter(graphs.values())).x.shape[1]),
        rw_dim + dg_dim,
        arguments,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-4)),
    )
    epochs = int(config["training"].get("epochs", 100))
    history = []
    statistics = {"mean_g": 0.0, "std_g": 1.0, "mean_n": 0.0, "std_n": 1.0}
    training_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        graph_values: list[float] = []
        node_values: list[float] = []
        total_loss = 0.0
        total_graphs = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            if mamba:
                outputs = model(
                    data,
                    data.x,
                    data.x_s,
                    data.edge_index,
                    data.batch,
                    data.num_graphs,
                    arguments,
                )
            else:
                outputs = model(
                    data.x,
                    data.x_s,
                    data.edge_index,
                    data.batch,
                    data.num_graphs,
                )
            graph_loss = model.calc_loss_g(outputs[0], outputs[1])
            node_loss = model.calc_loss_n(outputs[2], outputs[3], data.batch)
            if arguments.is_adaptive and epoch > 1:
                weight_g = statistics["std_g"] ** arguments.alpha
                weight_n = statistics["std_n"] ** arguments.alpha
                normalizer = max((weight_g + weight_n) / 2, 1e-12)
                weight_g, weight_n = weight_g / normalizer, weight_n / normalizer
            else:
                weight_g = weight_n = 1.0
            loss = weight_g * graph_loss.mean() + weight_n * node_loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * data.num_graphs
            total_graphs += data.num_graphs
            graph_values.extend(graph_loss.detach().cpu().tolist())
            node_values.extend(node_loss.detach().cpu().tolist())
        statistics = {
            "mean_g": float(np.mean(graph_values)),
            "std_g": max(float(np.std(graph_values)), 1e-12),
            "mean_n": float(np.mean(node_values)),
            "std_n": max(float(np.std(node_values)), 1e-12),
        }
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(total_graphs, 1),
                "epoch_seconds": time.perf_counter() - epoch_start,
                **statistics,
            }
        )
        print(
            f"[{official_name}-fair] epoch={epoch} "
            f"loss={history[-1]['loss']:.6f}",
            flush=True,
        )

    training_seconds = time.perf_counter() - training_start
    train_labels, train_scores, _ = _score_loader(
        model, train_eval_loader, device, arguments, mamba, statistics
    )
    threshold = normal_score_threshold(
        train_scores,
        float(config["evaluation"].get("threshold_quantile", 0.99)),
    )
    labels, scores, graph_ids = _score_loader(
        model, test_loader, device, arguments, mamba, statistics
    )
    metrics = anomaly_metrics(
        labels,
        scores,
        threshold=threshold,
        alert_fraction=float(config["evaluation"].get("alert_fraction", 0.01)),
        target_fpr=float(config["evaluation"].get("target_fpr", 0.01)),
    )
    output = (
        Path(config["output"].get("results_root", "artifacts/results"))
        / config["dataset"]["name"]
        / f"{official_name}-fair"
        / f"seed_{seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    pd.DataFrame(
        {"graph_id": graph_ids, "label": labels, "score": scores}
    ).to_csv(output / "test_predictions.csv", index=False)
    summary = {
        **metrics,
        "dataset": config["dataset"]["name"],
        "variant": f"{official_name}-fair",
        "seed": seed,
        "epochs_completed": epochs,
        "training_seconds": training_seconds,
        "parameters": parameter_count(model),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "device": str(device),
        "checkpoint_selection": "fixed_epoch",
        "threshold_source": "normal_train_scores",
        "official_source": str(Path(external_root) / official_name),
        "train_graphs": len(train_labels),
    }
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
