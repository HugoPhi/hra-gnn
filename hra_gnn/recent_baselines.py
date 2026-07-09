from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import load_dataset, load_provided_splits, make_splits
from .graph import batch_graphs
from .metrics import anomaly_metrics, normal_score_threshold
from .utils import parameter_count, resolve_device, seed_everything


def _official_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official model from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _official_module_with_path(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    try:
        return _official_module(path, name)
    finally:
        sys.path.pop(0)


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
        adjacency = to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes)
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
            graph.XLX_s = torch.diag(graph.x_s.T @ laplacian @ graph.x_s).unsqueeze(0)
        graphs[index] = graph
    return graphs


def _limited_splits(dataset, splits, maximum, *, seed: int):
    if maximum is None:
        return splits
    rng = np.random.default_rng(seed)
    limited = {}
    for split, indices in splits.items():
        by_label: dict[int, list[int]] = {}
        for index in indices:
            by_label.setdefault(int(dataset[index].label), []).append(index)
        if not by_label:
            limited[split] = []
            continue
        if len(by_label) == 1:
            candidates = np.asarray(indices, dtype=np.int64)
            rng.shuffle(candidates)
            limited[split] = candidates[:maximum].tolist()
            continue
        per_label = max(1, maximum // len(by_label))
        selected = []
        for label in sorted(by_label):
            candidates = np.asarray(by_label[label], dtype=np.int64)
            rng.shuffle(candidates)
            selected.extend(candidates[:per_label].tolist())
        rng.shuffle(selected)
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


def _index_batches(indices: list[int], batch_size: int, *, shuffle: bool) -> list[list[int]]:
    values = np.asarray(indices, dtype=np.int64)
    if shuffle:
        np.random.shuffle(values)
    return [
        values[start : start + batch_size].tolist()
        for start in range(0, len(values), batch_size)
    ]


def _score_audit(labels, scores, normal_scores):
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    normal_array = np.asarray(normal_scores, dtype=np.float64)

    def median_for(label):
        selected = scores_array[labels_array == label]
        return float(np.median(selected)) if selected.size else float("nan")

    inverse_auc = anomaly_metrics(labels, (-scores_array).tolist())["auc"]
    return {
        "score_direction": "higher_is_more_anomalous",
        "inverse_auc_diagnostic": inverse_auc,
        "test_normal_score_median": median_for(0),
        "test_anomaly_score_median": median_for(1),
        "normal_calibration_score_median": float(np.median(normal_array)),
        "normal_calibration_score_q95": float(np.quantile(normal_array, 0.95)),
    }


def _save_fair_checkpoint(
    output: Path,
    *,
    config: dict[str, Any],
    seed: int,
    epoch: int,
    model_states: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "seed": seed,
        "config": config,
        "model_states": model_states,
        "metadata": metadata or {},
    }
    torch.save(payload, checkpoint_dir / "best.pt")
    torch.save(payload, checkpoint_dir / "last.pt")


def _contrastive_losses(outputs, batch, temperature=0.2):
    from torch_geometric.nn import global_mean_pool

    def similarity(left, right):
        left = torch.nn.functional.normalize(left, dim=1, eps=1e-12)
        right = torch.nn.functional.normalize(right, dim=1, eps=1e-12)
        return torch.exp((left @ right.T / temperature).clamp(max=50.0))

    graph_similarity = similarity(outputs[0], outputs[1])
    graph_positive = graph_similarity.diag()
    graph_negative_columns = (graph_similarity.sum(dim=0) - graph_positive).clamp_min(
        1e-12
    )
    graph_negative_rows = (graph_similarity.sum(dim=1) - graph_positive).clamp_min(
        1e-12
    )
    graph_loss = -0.5 * (
        torch.log(graph_positive / graph_negative_columns + 1e-12)
        + torch.log(graph_positive / graph_negative_rows + 1e-12)
    )

    node_similarity = similarity(outputs[2], outputs[3])
    same_graph = batch[:, None] == batch[None, :]
    node_similarity = node_similarity * same_graph
    node_positive = node_similarity.diag()
    node_negative_columns = (node_similarity.sum(dim=0) - node_positive).clamp_min(
        1e-12
    )
    node_negative_rows = (node_similarity.sum(dim=1) - node_positive).clamp_min(1e-12)
    node_loss = -0.5 * (
        torch.log(node_positive / node_negative_columns + 1e-12)
        + torch.log(node_positive / node_negative_rows + 1e-12)
    )
    return graph_loss, global_mean_pool(node_loss, batch)


def _signet_loss(left, right, temperature=0.2):
    left = torch.nn.functional.normalize(left, dim=1, eps=1e-12)
    right = torch.nn.functional.normalize(right, dim=1, eps=1e-12)
    similarities = torch.exp((left @ right.T / temperature).clamp(max=50.0))
    positive = similarities.diag()
    negative_columns = (similarities.sum(dim=0) - positive).clamp_min(1e-12)
    negative_rows = (similarities.sum(dim=1) - positive).clamp_min(1e-12)
    return -0.5 * (
        torch.log(positive / negative_columns + 1e-12)
        + torch.log(positive / negative_rows + 1e-12)
    )


def _anchor_graph_loss(
    left,
    right,
    anchor_left,
    anchor_right,
    temperature=0.2,
):
    def similarity(first, second):
        first = torch.nn.functional.normalize(first, dim=1, eps=1e-12)
        second = torch.nn.functional.normalize(second, dim=1, eps=1e-12)
        return torch.exp((first @ second.T / temperature).clamp(max=50.0))

    positive = similarity(left, right).diag()
    negative_rows = similarity(left, anchor_right).sum(dim=1).clamp_min(1e-12)
    negative_columns = similarity(anchor_left, right).sum(dim=0).clamp_min(1e-12)
    return -0.5 * (
        torch.log(positive / negative_columns + 1e-12)
        + torch.log(positive / negative_rows + 1e-12)
    )


def _node_contrastive_loss(left, right, batch, temperature=0.2):
    from torch_geometric.nn import global_mean_pool

    left = torch.nn.functional.normalize(left, dim=1, eps=1e-12)
    right = torch.nn.functional.normalize(right, dim=1, eps=1e-12)
    similarities = torch.exp((left @ right.T / temperature).clamp(max=50.0))
    similarities = similarities * (batch[:, None] == batch[None, :])
    positive = similarities.diag()
    negative_columns = (similarities.sum(dim=0) - positive).clamp_min(1e-12)
    negative_rows = (similarities.sum(dim=1) - positive).clamp_min(1e-12)
    loss = -0.5 * (
        torch.log(positive / negative_columns + 1e-12)
        + torch.log(positive / negative_rows + 1e-12)
    )
    return global_mean_pool(loss, batch)


class NativeGraphEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_node_types: int,
        num_edge_types: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.self_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.message_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph) -> torch.Tensor:
        h = self.input_projection(graph.x.float()) + self.node_type_embedding(
            graph.node_type.long()
        )
        for self_layer, message_layer in zip(self.self_layers, self.message_layers):
            src, dst = graph.edge_index.long()
            edge_context = self.edge_type_embedding(graph.edge_type.long())
            messages = message_layer(h[src] + edge_context)
            aggregated = torch.zeros_like(h)
            aggregated.index_add_(0, dst, messages)
            degree = torch.zeros(h.shape[0], 1, device=h.device, dtype=h.dtype)
            degree.index_add_(0, dst, torch.ones_like(messages[:, :1]))
            aggregated = aggregated / degree.clamp_min(1.0)
            h = torch.relu(self_layer(h) + aggregated)
            h = self.dropout(h)
        batch = graph.batch
        if batch is None:
            batch = torch.zeros(h.shape[0], device=h.device, dtype=torch.long)
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
        mean_pool = torch.zeros(graph_count, h.shape[1], device=h.device, dtype=h.dtype)
        mean_pool.index_add_(0, batch, h)
        counts = torch.zeros(graph_count, 1, device=h.device, dtype=h.dtype)
        counts.index_add_(0, batch, torch.ones_like(h[:, :1]))
        mean_pool = mean_pool / counts.clamp_min(1.0)
        max_pool = torch.full_like(mean_pool, -torch.inf)
        max_pool.scatter_reduce_(
            0,
            batch[:, None].expand_as(h),
            h,
            reduce="amax",
            include_self=True,
        )
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return self.output_projection(torch.cat([mean_pool, max_pool], dim=1))


class HimNetNative(nn.Module):
    def __init__(self, encoder: NativeGraphEncoder, memory_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        output_dim = encoder.output_projection[-1].out_features
        self.memory = nn.Parameter(torch.randn(memory_size, output_dim) * 0.02)

    def forward(self, graph) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(graph)
        attention = torch.softmax(
            embedding @ self.memory.T / max(embedding.shape[1] ** 0.5, 1.0),
            dim=1,
        )
        reconstructed = attention @ self.memory
        return embedding, reconstructed

    def loss_and_score(self, graph) -> tuple[torch.Tensor, torch.Tensor]:
        embedding, reconstructed = self(graph)
        reconstruction = (embedding - reconstructed).square().mean(dim=1)
        nearest = torch.cdist(embedding, self.memory).min(dim=1).values.square()
        score = reconstruction + 0.1 * nearest
        return score.mean(), score


class GLADProNative(nn.Module):
    def __init__(self, encoder: NativeGraphEncoder, prototype_count: int) -> None:
        super().__init__()
        self.encoder = encoder
        output_dim = encoder.output_projection[-1].out_features
        self.prototypes = nn.Parameter(torch.randn(prototype_count, output_dim) * 0.02)

    def loss_and_score(self, graph) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(graph)
        distances = torch.cdist(embedding, self.prototypes).square()
        assignment = torch.softmax(-distances, dim=1)
        min_distance = distances.min(dim=1).values
        entropy = -(assignment * (assignment + 1e-12).log()).sum(dim=1)
        if self.prototypes.shape[0] > 1:
            prototype_distance = torch.pdist(self.prototypes).square().mean()
            diversity = 1.0 / prototype_distance.clamp_min(1e-6)
        else:
            diversity = torch.zeros((), device=embedding.device)
        loss = min_distance.mean() + 0.02 * entropy.mean() + 0.01 * diversity
        score = min_distance + 0.05 * entropy
        return loss, score


class MssGADNative(nn.Module):
    def __init__(
        self,
        encoder: NativeGraphEncoder,
        *,
        spaces: int,
        space_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        output_dim = encoder.output_projection[-1].out_features
        self.projections = nn.ModuleList(
            nn.Sequential(nn.Linear(output_dim, space_dim), nn.ReLU(), nn.Linear(space_dim, space_dim))
            for _ in range(spaces)
        )
        self.register_buffer("centers", torch.zeros(spaces, space_dim))
        self.register_buffer("centers_initialized", torch.tensor(False))

    @torch.no_grad()
    def initialize_centers(self, graph) -> None:
        embedding = self.encoder(graph)
        centers = [projection(embedding).mean(dim=0) for projection in self.projections]
        self.centers.copy_(torch.stack(centers))
        self.centers_initialized.fill_(True)

    def loss_and_score(self, graph) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(graph)
        distances = []
        projected = []
        for index, projection in enumerate(self.projections):
            value = projection(embedding)
            projected.append(torch.nn.functional.normalize(value, dim=1, eps=1e-12))
            distances.append((value - self.centers[index]).square().mean(dim=1))
        stacked = torch.stack(distances, dim=1)
        separation = torch.zeros((), device=embedding.device)
        for left in range(len(projected)):
            for right in range(left + 1, len(projected)):
                separation = separation + (projected[left] * projected[right]).sum(dim=1).abs().mean()
        loss = stacked.mean() + 0.05 * separation
        return loss, stacked.mean(dim=1)


def _native_model(config: dict[str, Any], architecture: str) -> nn.Module:
    baseline = config.get("recent_baseline", {})
    encoder = NativeGraphEncoder(
        input_dim=int(config["dataset"]["feature_dim"]),
        hidden_dim=int(baseline.get("hidden_dim", 64)),
        output_dim=int(baseline.get("output_dim", 64)),
        num_node_types=int(config["dataset"]["num_node_types"]),
        num_edge_types=int(config["dataset"]["num_edge_types"]),
        num_layers=int(baseline.get("num_layers", 3)),
        dropout=float(baseline.get("dropout", 0.1)),
    )
    if architecture == "himnet":
        return HimNetNative(encoder, int(baseline.get("memory_size", 16)))
    if architecture == "gladpro":
        return GLADProNative(encoder, int(baseline.get("prototype_count", 8)))
    if architecture == "mssgad":
        return MssGADNative(
            encoder,
            spaces=int(baseline.get("spaces", 3)),
            space_dim=int(baseline.get("space_dim", 32)),
        )
    raise ValueError(f"Unsupported native graph baseline: {architecture}")


def run_native_graph_fair(
    config: dict[str, Any],
    *,
    architecture: str,
) -> dict[str, Any]:
    names = {
        "himnet": "HimNet",
        "gladpro": "GLADPro",
        "mssgad": "MssGAD",
    }
    if architecture not in names:
        raise ValueError(f"Unsupported native graph baseline: {architecture}")
    baseline = config.get("recent_baseline", {})
    seed = int(config["training"].get("seed", 42))
    seed_everything(seed)
    device = resolve_device(config["training"].get("device", "auto"))
    dataset = load_dataset(config["dataset"])
    splits = _limited_splits(
        dataset,
        _splits(config, dataset),
        baseline.get("max_graphs_per_split"),
        seed=seed,
    )
    selected = sorted({index for values in splits.values() for index in values})
    graphs = {index: dataset[index] for index in selected}
    model = _native_model(config, architecture).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-6)),
    )
    batch_size = int(config["training"].get("batch_size", 32))
    epochs = int(config["training"].get("epochs", 100))
    if isinstance(model, MssGADNative):
        init_indices = splits["train"][: min(len(splits["train"]), max(batch_size, 2))]
        model.initialize_centers(batch_graphs([graphs[index] for index in init_indices]).to(device))
    history = []
    training_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        total = 0.0
        count = 0
        for batch in _index_batches(splits["train"], batch_size, shuffle=True):
            graph = batch_graphs([graphs[index] for index in batch]).to(device)
            optimizer.zero_grad()
            loss, _ = model.loss_and_score(graph)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        history.append(
            {
                "epoch": epoch,
                "loss": total / max(count, 1),
                "epoch_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"[{names[architecture]}-fair] epoch={epoch} loss={history[-1]['loss']:.6f}",
            flush=True,
        )

    def score(indices: list[int]) -> tuple[list[int], list[float], list[int]]:
        labels, scores, graph_ids = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in _index_batches(
                indices,
                int(config["evaluation"].get("batch_size", batch_size)),
                shuffle=False,
            ):
                graph = batch_graphs([graphs[index] for index in batch]).to(device)
                _, values = model.loss_and_score(graph)
                scores.extend(values.detach().cpu().tolist())
                labels.extend([graphs[index].label for index in batch])
                graph_ids.extend([graphs[index].graph_id for index in batch])
        return labels, scores, graph_ids

    training_seconds = time.perf_counter() - training_start
    train_labels, train_scores, train_graph_ids = score(splits["train"])
    threshold = normal_score_threshold(
        train_scores,
        float(config["evaluation"].get("threshold_quantile", 0.99)),
    )
    labels, scores, graph_ids = score(splits["test"])
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
        / f"{names[architecture]}-fair"
        / f"seed_{seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    pd.DataFrame(
        {"graph_id": train_graph_ids, "label": train_labels, "score": train_scores}
    ).to_csv(output / "normal_calibration_predictions.csv", index=False)
    pd.DataFrame({"graph_id": graph_ids, "label": labels, "score": scores}).to_csv(
        output / "test_predictions.csv", index=False
    )
    summary = {
        **metrics,
        **_score_audit(labels, scores, train_scores),
        "dataset": config["dataset"]["name"],
        "variant": f"{names[architecture]}-fair",
        "experimental_stage": baseline.get("experimental_stage", "standalone"),
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
        "scoring_protocol": f"{architecture}_native_graph_one_class",
        "train_graphs": len(train_scores),
        "subset_sampling": "seeded_stratified_random",
    }
    _save_fair_checkpoint(
        output,
        config=config,
        seed=seed,
        epoch=epochs,
        model_states={"model": model.state_dict()},
        metadata={"architecture": architecture},
    )
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _dual_view_outputs(model, data, args, mamba):
    if mamba:
        return model(
            data,
            data.x,
            data.x_s,
            data.edge_index,
            data.batch,
            data.num_graphs,
            args,
        )
    return model(
        data.x,
        data.x_s,
        data.edge_index,
        data.batch,
        data.num_graphs,
    )


def _dual_view_anchor_bank(model, loader, device, args, mamba):
    left, right = [], []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = _dual_view_outputs(model, data, args, mamba)
            left.append(outputs[0].detach())
            right.append(outputs[1].detach())
    return torch.cat(left, dim=0), torch.cat(right, dim=0)


def _score_loader(
    model,
    loader,
    device,
    args,
    mamba,
    statistics,
    anchor_bank,
):
    labels: list[int] = []
    scores: list[float] = []
    graph_ids: list[int] = []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = _dual_view_outputs(model, data, args, mamba)
            graph_loss = _anchor_graph_loss(
                outputs[0], outputs[1], anchor_bank[0], anchor_bank[1]
            )
            node_loss = _node_contrastive_loss(outputs[2], outputs[3], data.batch)
            if args.is_adaptive:
                score = (graph_loss - statistics["mean_g"]) / statistics["std_g"] + (
                    node_loss - statistics["mean_n"]
                ) / statistics["std_n"]
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
        seed=seed,
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
            graph_loss, node_loss = _contrastive_losses(outputs, data.batch)
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
            f"[{official_name}-fair] epoch={epoch} loss={history[-1]['loss']:.6f}",
            flush=True,
        )

    training_seconds = time.perf_counter() - training_start
    train_indices = dataset_splits["train"]
    anchor_count = min(
        int(baseline.get("reference_graphs", 64)),
        max(2, len(train_indices) // 2),
    )
    calibration_indices = train_indices[anchor_count:]
    if anchor_count < 2 or len(calibration_indices) < 2:
        raise ValueError(
            "Fixed-reference scoring needs at least two reference and "
            "two calibration normal graphs"
        )
    anchor_indices = train_indices[:anchor_count]
    anchor_loader = _evaluation_loader(
        [graphs[index] for index in anchor_indices],
        int(config["evaluation"].get("batch_size", batch_size)),
    )
    train_eval_loader = _evaluation_loader(
        [graphs[index] for index in calibration_indices], batch_size
    )
    anchor_bank = _dual_view_anchor_bank(model, anchor_loader, device, arguments, mamba)
    train_labels, train_scores, train_graph_ids = _score_loader(
        model,
        train_eval_loader,
        device,
        arguments,
        mamba,
        statistics,
        anchor_bank,
    )
    threshold = normal_score_threshold(
        train_scores,
        float(config["evaluation"].get("threshold_quantile", 0.99)),
    )
    labels, scores, graph_ids = _score_loader(
        model,
        test_loader,
        device,
        arguments,
        mamba,
        statistics,
        anchor_bank,
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
        {
            "graph_id": train_graph_ids,
            "label": train_labels,
            "score": train_scores,
        }
    ).to_csv(output / "normal_calibration_predictions.csv", index=False)
    pd.DataFrame({"graph_id": graph_ids, "label": labels, "score": scores}).to_csv(
        output / "test_predictions.csv", index=False
    )
    summary = {
        **metrics,
        **_score_audit(labels, scores, train_scores),
        "dataset": config["dataset"]["name"],
        "variant": f"{official_name}-fair",
        "experimental_stage": baseline.get("experimental_stage", "standalone"),
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
        "scoring_protocol": "fixed_normal_reference_bank",
        "reference_graphs": anchor_count,
        "normal_calibration_graphs": len(train_scores),
        "subset_sampling": "seeded_stratified_random",
        "official_source": str(Path(external_root) / official_name),
        "train_graphs": len(train_labels),
        "numerically_stabilized": True,
    }
    _save_fair_checkpoint(
        output,
        config=config,
        seed=seed,
        epoch=epochs,
        model_states={"model": model.state_dict()},
        metadata={
            "official_name": official_name,
            "statistics": statistics,
            "anchor_count": anchor_count,
        },
    )
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def run_signet_fair(
    config: dict[str, Any], *, external_root: str | Path = "external"
) -> dict[str, Any]:
    try:
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
    except ImportError as exc:
        raise RuntimeError("SIGNET requires requirements-baselines.txt") from exc

    baseline = config.get("recent_baseline", {})
    seed = int(config["training"].get("seed", 42))
    seed_everything(seed)
    device = resolve_device(config["training"].get("device", "auto"))
    dataset = load_dataset(config["dataset"])
    dataset_splits = _limited_splits(
        dataset,
        _splits(config, dataset),
        baseline.get("max_graphs_per_split"),
        seed=seed,
    )
    selected = sorted({index for values in dataset_splits.values() for index in values})
    graphs = {}
    for index in selected:
        source = dataset[index]
        graphs[index] = Data(
            x=source.x.float(),
            edge_index=source.edge_index.long(),
            edge_attr=None,
            edge_type=source.edge_type.long(),
            y=torch.tensor([source.label], dtype=torch.long),
            graph_id=torch.tensor([source.graph_id], dtype=torch.long),
        )
    batch_size = int(config["training"].get("batch_size", 32))
    if batch_size < 2:
        raise ValueError("SIGNET needs batch_size >= 2 for contrastive negatives")
    train_graphs = [graphs[index] for index in dataset_splits["train"]]
    test_graphs = [graphs[index] for index in dataset_splits["test"]]
    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = _evaluation_loader(
        test_graphs,
        int(config["evaluation"].get("batch_size", batch_size)),
    )

    module = _official_module_with_path(
        Path(external_root) / "SIGNET" / "main.py",
        f"official_signet_{seed}",
    )
    arguments = SimpleNamespace(
        hidden_dim=int(baseline.get("hidden_dim", 16)),
        encoder_layers=int(baseline.get("num_layers", 5)),
        pooling=baseline.get("pooling", "add"),
        readout=baseline.get("readout", "concat"),
        explainer_model=baseline.get("explainer_model", "gin"),
        explainer_hidden_dim=int(baseline.get("explainer_hidden_dim", 8)),
        explainer_layers=int(baseline.get("explainer_layers", 5)),
        explainer_readout=baseline.get("explainer_readout", "add"),
    )
    model = module.SIGNET(int(train_graphs[0].x.shape[1]), 0, arguments, device).to(
        device
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-4)),
    )
    epochs = int(config["training"].get("epochs", 100))
    history = []
    training_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        total = 0.0
        count = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            left, right, _, _ = model(data)
            loss = _signet_loss(left, right).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * data.num_graphs
            count += data.num_graphs
        history.append(
            {
                "epoch": epoch,
                "loss": total / max(count, 1),
                "epoch_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"[SIGNET-fair] epoch={epoch} loss={history[-1]['loss']:.6f}",
            flush=True,
        )

    def encode_reference(loader):
        left_values, right_values = [], []
        model.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                left, right, _, _ = model(data)
                left_values.append(left.detach())
                right_values.append(right.detach())
        return torch.cat(left_values), torch.cat(right_values)

    anchor_count = min(
        int(baseline.get("reference_graphs", 64)),
        max(2, len(train_graphs) // 2),
    )
    calibration_graphs = train_graphs[anchor_count:]
    if anchor_count < 2 or len(calibration_graphs) < 2:
        raise ValueError(
            "Fixed-reference scoring needs at least two reference and "
            "two calibration normal graphs"
        )
    anchor_loader = _evaluation_loader(
        train_graphs[:anchor_count],
        int(config["evaluation"].get("batch_size", batch_size)),
    )
    train_eval_loader = _evaluation_loader(calibration_graphs, batch_size)
    anchor_bank = encode_reference(anchor_loader)

    def score(loader):
        labels, scores, graph_ids = [], [], []
        model.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                left, right, _, _ = model(data)
                values = _anchor_graph_loss(left, right, anchor_bank[0], anchor_bank[1])
                labels.extend(data.y.reshape(-1).cpu().tolist())
                scores.extend(values.cpu().tolist())
                graph_ids.extend(data.graph_id.reshape(-1).cpu().tolist())
        return labels, scores, graph_ids

    training_seconds = time.perf_counter() - training_start
    train_labels, train_scores, train_graph_ids = score(train_eval_loader)
    threshold = normal_score_threshold(
        train_scores,
        float(config["evaluation"].get("threshold_quantile", 0.99)),
    )
    labels, scores, graph_ids = score(test_loader)
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
        / "SIGNET-fair"
        / f"seed_{seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    pd.DataFrame(
        {
            "graph_id": train_graph_ids,
            "label": train_labels,
            "score": train_scores,
        }
    ).to_csv(output / "normal_calibration_predictions.csv", index=False)
    pd.DataFrame({"graph_id": graph_ids, "label": labels, "score": scores}).to_csv(
        output / "test_predictions.csv", index=False
    )
    summary = {
        **metrics,
        **_score_audit(labels, scores, train_scores),
        "dataset": config["dataset"]["name"],
        "variant": "SIGNET-fair",
        "experimental_stage": baseline.get("experimental_stage", "standalone"),
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
        "scoring_protocol": "fixed_normal_reference_bank",
        "reference_graphs": anchor_count,
        "normal_calibration_graphs": len(train_scores),
        "subset_sampling": "seeded_stratified_random",
        "official_source": str(Path(external_root) / "SIGNET"),
        "train_graphs": len(train_labels),
        "numerically_stabilized": True,
    }
    _save_fair_checkpoint(
        output,
        config=config,
        seed=seed,
        epoch=epochs,
        model_states={"model": model.state_dict()},
        metadata={"official_name": "SIGNET", "anchor_count": anchor_count},
    )
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def run_muse_fair(
    config: dict[str, Any], *, external_root: str | Path = "external"
) -> dict[str, Any]:
    try:
        from torch_geometric.data import Data
        from torch_geometric.utils import to_dense_adj
    except ImportError as exc:
        raise RuntimeError("MUSE requires requirements-baselines.txt") from exc

    baseline = config.get("recent_baseline", {})
    seed = int(config["training"].get("seed", 42))
    seed_everything(seed)
    device = resolve_device(config["training"].get("device", "auto"))
    dataset = load_dataset(config["dataset"])
    splits = _limited_splits(
        dataset,
        _splits(config, dataset),
        baseline.get("max_graphs_per_split"),
        seed=seed,
    )
    selected = sorted({index for values in splits.values() for index in values})
    index_map = {dataset_index: local for local, dataset_index in enumerate(selected)}
    local_splits = {
        split: [index_map[index] for index in indices]
        for split, indices in splits.items()
    }
    graphs = []
    graph_ids = []
    max_dense_nodes = int(baseline.get("muse_max_dense_nodes", 2048))
    for index in selected:
        source = dataset[index]
        if source.num_nodes > max_dense_nodes:
            raise RuntimeError(
                f"MUSE dense adjacency limit exceeded: graph {source.graph_id} "
                f"has {source.num_nodes} nodes, limit is {max_dense_nodes}"
            )
        graphs.append(
            Data(
                x=source.x.float(),
                edge_index=source.edge_index.long(),
                y=torch.tensor([source.label], dtype=torch.long),
            )
        )
        graph_ids.append(source.graph_id)

    official_root = Path(external_root) / "MUSE"
    gnns = _official_module_with_path(
        official_root / "GNNs.py", f"official_muse_gnns_{seed}"
    )
    source_module = _official_module_with_path(
        official_root / "src.py", f"official_muse_src_{seed}"
    )
    hidden_dim = int(baseline.get("hidden_dim", 32))
    layers = int(baseline.get("num_layers", 3))
    representation_epochs = int(
        baseline.get(
            "muse_representation_epochs", config["training"].get("epochs", 100)
        )
    )
    classifier_epochs = int(baseline.get("muse_classifier_epochs", 100))
    representation_batch = int(baseline.get("muse_representation_batch_size", 16))
    model = gnns.GIN(
        num_features=int(graphs[0].x.shape[1]),
        num_classes=1,
        hidden_units=hidden_dim,
        num_layers=layers,
        dropout=0.3,
        mlp_layers=2,
        train_eps=False,
    ).to(device)
    edge_decoder = gnns.MLP_Decoder(layers * hidden_dim, hidden_dim, hidden_dim).to(
        device
    )
    feature_decoder = gnns.MLP_Decoder(
        layers * hidden_dim, hidden_dim, int(graphs[0].x.shape[1])
    ).to(device)

    adjacency_labels = []
    positive_weights = []
    for graph in graphs:
        adjacency = to_dense_adj(graph.edge_index)[0]
        adjacency.fill_diagonal_(1.0)
        flattened = adjacency.flatten().to(device)
        positives = flattened.sum().clamp_min(1.0)
        positive_weights.append(
            ((flattened.numel() - positives) / positives).clamp_min(1e-6)
        )
        adjacency_labels.append(flattened)

    trainer = source_module.MUSE_representation_learning(
        datasets=graphs,
        device=device,
        labels=adjacency_labels,
        labels_pos_weights=positive_weights,
    )
    training_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    losses, parameters = trainer.train(
        model=model,
        feature_head=feature_decoder,
        edge_head=edge_decoder,
        train_idxs=local_splits["train"],
        lr=float(config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-6)),
        epochs=representation_epochs,
        saving_interval=representation_epochs,
        batch_size=representation_batch,
        return_loss=True,
        seed=seed,
    )
    if not parameters:
        raise RuntimeError("MUSE representation training produced no checkpoint")
    extractor = source_module.MUSE_oneclass_classification(
        model=model,
        feature_encoder=feature_decoder,
        edge_encoder=edge_decoder,
        datasets=graphs,
        device=device,
        labels=adjacency_labels,
        pos_weights=positive_weights,
        B_size=int(baseline.get("muse_extraction_batch_size", 16)),
    )
    extractor.obtain_error_representations(parameters[-1], local_splits["train"])
    features = extractor.TX
    classifier = gnns.AutoEncoderOneclassClassifier(
        in_dim=4,
        hid_dim=int(baseline.get("muse_classifier_hidden_dim", 64)),
        n_layers=3,
        drop_p=0.0,
    ).to(device)
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=float(baseline.get("muse_classifier_lr", 1e-3)),
        weight_decay=float(baseline.get("muse_classifier_weight_decay", 1e-4)),
    )
    train_index = torch.tensor(local_splits["train"], device=device)
    classifier_history = []
    for epoch in range(1, classifier_epochs + 1):
        classifier.train()
        optimizer.zero_grad()
        reconstructed = classifier(features[train_index])
        loss = torch.sqrt(
            (reconstructed - features[train_index]).square().sum(dim=1)
        ).mean()
        loss.backward()
        optimizer.step()
        classifier_history.append(float(loss.detach()))
        print(f"[MUSE-fair] classifier_epoch={epoch} loss={loss:.6f}", flush=True)

    classifier.eval()
    with torch.no_grad():
        train_reconstructed = classifier(features[train_index])
        scale = train_reconstructed.std(dim=0).clamp_min(1e-6)

        def score(indices):
            index_tensor = torch.tensor(indices, device=device)
            reconstructed = classifier(features[index_tensor])
            return (
                ((reconstructed - features[index_tensor]).square() / scale)
                .sum(dim=1)
                .cpu()
                .tolist()
            )

        train_scores = score(local_splits["train"])
        test_scores = score(local_splits["test"])
    threshold = normal_score_threshold(
        train_scores,
        float(config["evaluation"].get("threshold_quantile", 0.99)),
    )
    test_labels = [dataset[index].label for index in splits["test"]]
    metrics = anomaly_metrics(
        test_labels,
        test_scores,
        threshold=threshold,
        alert_fraction=float(config["evaluation"].get("alert_fraction", 0.01)),
        target_fpr=float(config["evaluation"].get("target_fpr", 0.01)),
    )
    output = (
        Path(config["output"].get("results_root", "artifacts/results"))
        / config["dataset"]["name"]
        / "MUSE-fair"
        / f"seed_{seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "epoch": range(1, representation_epochs + 1),
            "adjacency_loss": losses[0],
            "feature_loss": losses[1],
        }
    ).to_csv(output / "representation_history.csv", index=False)
    pd.DataFrame(
        {
            "epoch": range(1, classifier_epochs + 1),
            "loss": classifier_history,
        }
    ).to_csv(output / "classifier_history.csv", index=False)
    test_graph_ids = [dataset[index].graph_id for index in splits["test"]]
    pd.DataFrame(
        {
            "graph_id": test_graph_ids,
            "label": test_labels,
            "score": test_scores,
        }
    ).to_csv(output / "test_predictions.csv", index=False)
    summary = {
        **metrics,
        **_score_audit(test_labels, test_scores, train_scores),
        "dataset": config["dataset"]["name"],
        "variant": "MUSE-fair",
        "experimental_stage": baseline.get("experimental_stage", "standalone"),
        "seed": seed,
        "epochs_completed": representation_epochs,
        "classifier_epochs": classifier_epochs,
        "training_seconds": time.perf_counter() - training_start,
        "parameters": parameter_count(model)
        + parameter_count(feature_decoder)
        + parameter_count(edge_decoder)
        + parameter_count(classifier),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "device": str(device),
        "checkpoint_selection": "fixed_epoch",
        "threshold_source": "normal_train_scores",
        "scoring_protocol": "per_graph_reconstruction_error",
        "subset_sampling": "seeded_stratified_random",
        "official_source": str(official_root),
        "train_graphs": len(local_splits["train"]),
        "dense_adjacency_limit": max_dense_nodes,
    }
    _save_fair_checkpoint(
        output,
        config=config,
        seed=seed,
        epoch=representation_epochs,
        model_states={
            "representation": model.state_dict(),
            "feature_decoder": feature_decoder.state_dict(),
            "edge_decoder": edge_decoder.state_dict(),
            "classifier": classifier.state_dict(),
        },
        metadata={
            "official_name": "MUSE",
            "classifier_epochs": classifier_epochs,
            "local_splits": local_splits,
        },
    )
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
