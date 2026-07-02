from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .augment import HeterogeneousAugmentor
from .baselines import DeepTraLogBaseline, GLocalKDBaseline, HGTBaseline
from .config import save_resolved_config
from .data import load_dataset, load_provided_splits, make_splits
from .graph import batch_graphs
from .metrics import anomaly_metrics
from .model import HRAGNN
from .utils import parameter_count, resolve_device, seed_everything, write_json


def build_model(config: dict[str, Any]) -> nn.Module:
    dataset = config["dataset"]
    model = config["model"]
    common = dict(
        input_dim=dataset["feature_dim"],
        hidden_dim=model["hidden_dim"],
        output_dim=model["output_dim"],
        num_node_types=dataset["num_node_types"],
        num_layers=model.get("num_layers", 2),
    )
    architecture = model.get("architecture", "hra")
    if architecture == "hgt":
        return HGTBaseline(
            **common,
            num_edge_types=dataset["num_edge_types"],
            dropout=model.get("dropout", 0.1),
        )
    if architecture == "deeptralog":
        return DeepTraLogBaseline(**common)
    if architecture == "glocalkd":
        return GLocalKDBaseline(
            **common,
            num_edge_types=dataset["num_edge_types"],
            dropout=model.get("dropout", 0.1),
        )
    if architecture not in {"hra", "ochetgcn", "hrgcn"}:
        raise ValueError(f"Unsupported model architecture: {architecture}")
    return HRAGNN(
        **common,
        num_edge_types=dataset["num_edge_types"],
        relation_fusion=model.get("relation_fusion", "deviation_attention"),
        deviation_weight=model.get("deviation_weight", 1.0),
        attention_temperature=model.get("attention_temperature", 1.0),
        prototype_momentum=model.get("prototype_momentum", 0.9),
        prototype_min_scale=model.get("prototype_min_scale", 1e-3),
        readout=model.get("readout", "hybrid"),
        dropout=model.get("dropout", 0.1),
        score_ssl_weight=config["evaluation"].get("score_ssl_weight", 1.0),
        score_mode=config["evaluation"].get("score_mode", "paper_product"),
    )


def build_augmentor(config: dict[str, Any], seed: int) -> HeterogeneousAugmentor:
    dataset = config["dataset"]
    augmentation = config["augmentation"]
    return HeterogeneousAugmentor(
        num_node_types=dataset["num_node_types"],
        num_edge_types=dataset["num_edge_types"],
        edge_perturbation_rate=augmentation.get("edge_perturbation_rate", 0.1),
        edge_addition_rate=augmentation.get("edge_addition_rate", 0.1),
        node_type_swap_rate=augmentation.get("node_type_swap_rate", 0.1),
        edge_type_swap_rate=augmentation.get("edge_type_swap_rate", 0.1),
        methods=augmentation.get("methods"),
        preserve_observed_schema=augmentation.get("preserve_observed_schema", True),
        seed=seed,
    )


def result_directory(config: dict[str, Any], seed: int) -> Path:
    output = config["output"]
    dataset_name = config["dataset"]["name"]
    run_name = output.get("run_name", "hra_full")
    return (
        Path(output.get("results_root", "artifacts/results"))
        / dataset_name
        / run_name
        / f"seed_{seed}"
    )


class Trainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.seed = int(config["training"].get("seed", 42))
        seed_everything(self.seed)
        self.device = resolve_device(config["training"].get("device", "auto"))
        self.dataset = load_dataset(config["dataset"])
        split_config = config["dataset"].get("split", {})
        if split_config.get("mode", "generated") == "provided":
            self.splits = load_provided_splits(
                self.dataset,
                config["dataset"]["root"],
                train_file=split_config.get("train_file", "model_gid_list_train.txt"),
                validation_file=split_config.get("validation_file"),
                test_file=split_config.get("test_file", "model_gid_list_eval.txt"),
            )
        else:
            self.splits = make_splits(
                self.dataset,
                train_normal_ratio=split_config.get("train_normal_ratio", 0.6),
                validation_normal_ratio=split_config.get(
                    "validation_normal_ratio", 0.2
                ),
                validation_anomaly_ratio=split_config.get(
                    "validation_anomaly_ratio", 0.0
                ),
                seed=self.seed,
            )
        self.monitoring = config.get("monitoring", {})
        self._configure_monitor_splits()
        self.model = build_model(config).to(self.device)
        self.augmentor = build_augmentor(config, self.seed)
        self.output_dir = result_directory(config, self.seed)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_resolved_config(config, self.output_dir / "config.yaml")
        write_json(self.splits, self.output_dir / "splits.json")

        training = config["training"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=training.get("learning_rate", 1e-3),
            weight_decay=training.get("weight_decay", 1e-6),
        )
        self.ssl_enabled = bool(config.get("ssl", {}).get("enabled", True))
        self.ssl_loss_weight = float(config.get("ssl", {}).get("loss_weight", 0.1))
        self.bce = nn.BCEWithLogitsLoss()
        self.writer = self._build_summary_writer()

    def _label_for_index(self, index: int) -> int:
        if hasattr(self.dataset, "labels") and hasattr(self.dataset, "graph_ids"):
            graph_id = self.dataset.graph_ids[index]
            return int(self.dataset.labels[graph_id])
        return int(self.dataset[index].label)

    def _configure_monitor_splits(self) -> None:
        if not bool(self.monitoring.get("enabled", False)):
            return
        if self.splits["validation"]:
            self.splits["monitor_validation"] = list(self.splits["validation"])
            self.splits["monitor_test"] = list(self.splits["test"])
            return

        rng = np.random.default_rng(self.seed)
        fraction = float(self.monitoring.get("validation_fraction_from_test", 0.5))
        validation: list[int] = []
        test: list[int] = []
        by_label: dict[int, list[int]] = {}
        for index in self.splits["test"]:
            by_label.setdefault(self._label_for_index(index), []).append(index)
        for values in by_label.values():
            shuffled = np.asarray(values)
            rng.shuffle(shuffled)
            cut = max(1, min(len(shuffled) - 1, round(len(shuffled) * fraction)))
            validation.extend(shuffled[:cut].tolist())
            test.extend(shuffled[cut:].tolist())
        rng.shuffle(validation)
        rng.shuffle(test)
        self.splits["monitor_validation"] = validation
        self.splits["monitor_test"] = test

    def _build_summary_writer(self) -> SummaryWriter | None:
        if not bool(self.monitoring.get("enabled", False)):
            return None
        root = Path(self.monitoring.get("log_dir", "artifacts/tensorboard"))
        dataset_name = self.config["dataset"]["name"]
        run_name = self.config["output"].get("run_name", "hra_full")
        self.tensorboard_prefix = dataset_name
        writer = SummaryWriter(
            root / dataset_name / run_name / f"seed_{self.seed}",
            flush_secs=int(self.monitoring.get("flush_seconds", 10)),
        )
        writer.add_custom_scalars(
            {
                dataset_name: {
                    "SVDD Loss: train/test": [
                        "Multiline",
                        [
                            f"{dataset_name}/Loss/train",
                            f"{dataset_name}/Loss/test",
                        ],
                    ],
                    "AUC: validation/test": [
                        "Multiline",
                        [
                            f"{dataset_name}/AUC/validation",
                            f"{dataset_name}/AUC/test",
                        ],
                    ],
                    "AP: validation/test": [
                        "Multiline",
                        [
                            f"{dataset_name}/AP/validation",
                            f"{dataset_name}/AP/test",
                        ],
                    ],
                }
            }
        )
        return writer

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _batches(self, indices: list[int], shuffle: bool) -> list[list[int]]:
        values = np.asarray(indices)
        if shuffle:
            np.random.shuffle(values)
        maximum = self.config["training"].get("max_train_graphs")
        if maximum is not None:
            values = values[: int(maximum)]
        batch_size = int(self.config["training"].get("batch_size", 8))
        return [
            values[start : start + batch_size].tolist()
            for start in range(0, len(values), batch_size)
        ]

    def _forward_original(
        self, indices: list[int], *, update_prototypes: bool
    ) -> tuple[Any, torch.Tensor]:
        if update_prototypes and hasattr(self.model, "begin_prototype_batch"):
            self.model.begin_prototype_batch()
        graph = batch_graphs([self.dataset[index] for index in indices]).to(self.device)
        output = self.model(graph, update_prototypes=update_prototypes)
        if update_prototypes and hasattr(self.model, "commit_prototype_batch"):
            self.model.commit_prototype_batch()
        embeddings = output.embedding
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        return output, embeddings

    def _train_batch(self, indices: list[int]) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        output, embeddings = self._forward_original(indices, update_prototypes=True)
        if isinstance(self.model, GLocalKDBaseline):
            distillation_loss = self.model.distillation_loss(output)
            distillation_loss.backward()
            self.optimizer.step()
            value = float(distillation_loss.detach().cpu())
            return {"loss": value, "svdd_loss": value, "ssl_loss": 0.0}
        if not bool(self.model.svdd_center_initialized):
            self.model.initialize_svdd_center(embeddings)

        svdd_loss = (embeddings - self.model.svdd_center.unsqueeze(0)).square().mean()
        ssl_loss = torch.zeros((), device=self.device)
        if self.ssl_enabled:
            original_logits = output.ssl_logit.reshape(-1)
            augmented = batch_graphs(
                [self.augmentor.augment(self.dataset[index]).graph for index in indices]
            ).to(self.device)
            augmented_logits = self.model(
                augmented, update_prototypes=False
            ).ssl_logit.reshape(-1)
            logits = torch.cat([original_logits, augmented_logits])
            labels = torch.cat(
                [
                    torch.ones_like(original_logits),
                    torch.zeros_like(original_logits),
                ]
            )
            ssl_loss = self.bce(logits, labels)

        total_loss = svdd_loss + self.ssl_loss_weight * ssl_loss
        total_loss.backward()
        gradient_clip = self.config["training"].get("gradient_clip")
        if gradient_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
        self.optimizer.step()
        return {
            "loss": float(total_loss.detach().cpu()),
            "svdd_loss": float(svdd_loss.detach().cpu()),
            "ssl_loss": float(ssl_loss.detach().cpu()),
        }

    @torch.no_grad()
    def evaluate(
        self,
        split: str,
        *,
        collect_diagnostics: bool = False,
        write_predictions: bool = False,
    ) -> dict[str, Any]:
        self.model.eval()
        if not self.splits[split]:
            return {
                "auc": math.nan,
                "ap": math.nan,
                "mean_svdd_loss": math.nan,
                "num_graphs": 0,
                "mean_inference_seconds": math.nan,
            }
        labels: list[int] = []
        scores: list[float] = []
        rows: list[dict[str, Any]] = []
        relation_rows: list[dict[str, Any]] = []
        elapsed: list[float] = []

        split_indices = self.splits[split]
        maximum = self.config["evaluation"].get("max_graphs")
        if maximum is not None:
            split_indices = split_indices[: int(maximum)]
        batch_size = int(
            self.config["evaluation"].get(
                "batch_size", self.config["training"].get("batch_size", 8)
            )
        )
        batches = [
            split_indices[start : start + batch_size]
            for start in range(0, len(split_indices), batch_size)
        ]
        for batch_indices in batches:
            graphs = [self.dataset[index] for index in batch_indices]
            graph = batch_graphs(graphs).to(self.device)
            self._sync()
            start = time.perf_counter()
            output = self.model(
                graph,
                update_prototypes=False,
                collect_diagnostics=False,
            )
            score_values = self.model.anomaly_score(output).reshape(-1)
            embedding_values = output.embedding
            if embedding_values.ndim == 1:
                embedding_values = embedding_values.unsqueeze(0)
            svdd_values = torch.full_like(score_values, math.nan)
            if hasattr(self.model, "svdd_center") and bool(
                getattr(self.model, "svdd_center_initialized", False)
            ):
                svdd_values = (
                    (embedding_values - self.model.svdd_center).square().mean(dim=1)
                )
            ssl_logits = output.ssl_logit.reshape(-1)
            ssl_anomaly_values = 1.0 - ssl_logits.sigmoid()
            gate_values = output.gate
            if gate_values.ndim == 1:
                gate_values = gate_values.unsqueeze(0)
            self._sync()
            batch_elapsed = time.perf_counter() - start
            elapsed.extend([batch_elapsed / len(graphs)] * len(graphs))
            score_cpu = score_values.detach().cpu().tolist()
            svdd_cpu = svdd_values.detach().cpu().tolist()
            ssl_cpu = ssl_anomaly_values.detach().cpu().tolist()
            probability_cpu = ssl_logits.sigmoid().detach().cpu().tolist()
            gate_cpu = gate_values.detach().mean(dim=1).cpu().tolist()
            for item, graph_item in enumerate(graphs):
                labels.append(graph_item.label)
                scores.append(float(score_cpu[item]))
                rows.append(
                    {
                        "graph_id": graph_item.graph_id,
                        "label": graph_item.label,
                        "score": scores[-1],
                        "svdd_score": float(svdd_cpu[item]),
                        "ssl_anomaly_score": float(ssl_cpu[item]),
                        "ssl_probability": float(probability_cpu[item]),
                        "gate_mean": float(gate_cpu[item]),
                    }
                )
            if collect_diagnostics:
                for graph_item in graphs:
                    diagnostic_output = self.model(
                        graph_item.to(self.device),
                        update_prototypes=False,
                        collect_diagnostics=True,
                    )
                    for (
                        relation_id,
                        diagnostics,
                    ) in diagnostic_output.relation_diagnostics.items():
                        relation_rows.append(
                            {
                                "graph_id": graph_item.graph_id,
                                "label": graph_item.label,
                                "relation_id": relation_id,
                                **diagnostics,
                            }
                        )

        metrics = anomaly_metrics(labels, scores)
        finite_svdd = [row["svdd_score"] for row in rows]
        if finite_svdd and not any(math.isnan(value) for value in finite_svdd):
            svdd_metrics = anomaly_metrics(labels, finite_svdd)
            metrics["svdd_auc"] = svdd_metrics["auc"]
            metrics["svdd_ap"] = svdd_metrics["ap"]
            metrics["mean_svdd_loss"] = float(np.mean(finite_svdd))
        else:
            metrics["mean_svdd_loss"] = math.nan
        ssl_metrics = anomaly_metrics(
            labels, [row["ssl_anomaly_score"] for row in rows]
        )
        metrics["ssl_auc"] = ssl_metrics["auc"]
        metrics["ssl_ap"] = ssl_metrics["ap"]
        metrics["num_graphs"] = len(labels)
        metrics["mean_inference_seconds"] = (
            float(np.mean(elapsed)) if elapsed else math.nan
        )
        if write_predictions:
            pd.DataFrame(rows).to_csv(
                self.output_dir / f"{split}_predictions.csv", index=False
            )
            if relation_rows:
                pd.DataFrame(relation_rows).to_csv(
                    self.output_dir / f"{split}_relations.csv", index=False
                )
        return metrics

    def _save_checkpoint(self, name: str, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config,
            },
            self.checkpoint_dir / name,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])

    def train(self) -> dict[str, Any]:
        history_path = self.output_dir / "history.csv"
        maximum_epochs = int(self.config["training"].get("epochs", 100))
        evaluate_every = int(self.config["training"].get("evaluate_every", 1))
        patience = int(self.config["training"].get("early_stopping_patience", 10))
        best_metric = -math.inf
        stale_epochs = 0
        history: list[dict[str, float]] = []
        process = psutil.Process()
        peak_rss = process.memory_info().rss
        training_start = time.perf_counter()

        for epoch in range(1, maximum_epochs + 1):
            self.model.train()
            epoch_start = time.perf_counter()
            batch_metrics = [
                self._train_batch(batch)
                for batch in self._batches(self.splits["train"], shuffle=True)
            ]
            peak_rss = max(peak_rss, process.memory_info().rss)
            row: dict[str, float] = {
                "epoch": float(epoch),
                "loss": float(np.mean([item["loss"] for item in batch_metrics])),
                "svdd_loss": float(
                    np.mean([item["svdd_loss"] for item in batch_metrics])
                ),
                "ssl_loss": float(
                    np.mean([item["ssl_loss"] for item in batch_metrics])
                ),
                "epoch_seconds": time.perf_counter() - epoch_start,
            }

            if epoch % evaluate_every == 0:
                validation = self.evaluate("validation")
                row["validation_auc"] = validation["auc"]
                row["validation_ap"] = validation["ap"]
                monitored = validation["auc"]
                if math.isnan(monitored):
                    monitored = -row["loss"]
                if monitored > best_metric:
                    best_metric = monitored
                    stale_epochs = 0
                    self._save_checkpoint("best.pt", epoch)
                else:
                    stale_epochs += 1
                if self.writer is not None:
                    monitor_validation = (
                        validation
                        if self.splits["monitor_validation"]
                        == self.splits["validation"]
                        else self.evaluate("monitor_validation")
                    )
                    monitor_test = self.evaluate("monitor_test")
                    row["monitor_validation_auc"] = monitor_validation["auc"]
                    row["monitor_validation_ap"] = monitor_validation["ap"]
                    row["monitor_test_auc"] = monitor_test["auc"]
                    row["monitor_test_ap"] = monitor_test["ap"]
                    row["monitor_test_svdd_loss"] = monitor_test["mean_svdd_loss"]
                    prefix = self.tensorboard_prefix
                    self.writer.add_scalar(
                        f"{prefix}/Loss/train", row["svdd_loss"], epoch
                    )
                    self.writer.add_scalar(
                        f"{prefix}/Loss/test",
                        monitor_test["mean_svdd_loss"],
                        epoch,
                    )
                    self.writer.add_scalar(
                        f"{prefix}/AUC/validation",
                        monitor_validation["auc"],
                        epoch,
                    )
                    self.writer.add_scalar(
                        f"{prefix}/AUC/test", monitor_test["auc"], epoch
                    )
                    self.writer.add_scalar(
                        f"{prefix}/AP/validation",
                        monitor_validation["ap"],
                        epoch,
                    )
                    self.writer.add_scalar(
                        f"{prefix}/AP/test", monitor_test["ap"], epoch
                    )
                    self.writer.flush()
            history.append(row)
            pd.DataFrame(history).to_csv(history_path, index=False)
            progress = (
                f"[{self.config['dataset']['name']}] epoch={epoch} "
                f"loss={row['loss']:.6f} svdd={row['svdd_loss']:.6f} "
                f"ssl={row['ssl_loss']:.6f} seconds={row['epoch_seconds']:.1f}"
            )
            if "monitor_test_auc" in row:
                progress += (
                    f" val_auc={row['monitor_validation_auc']:.4f}"
                    f" val_ap={row['monitor_validation_ap']:.4f}"
                    f" test_auc={row['monitor_test_auc']:.4f}"
                    f" test_ap={row['monitor_test_ap']:.4f}"
                )
            print(progress, flush=True)
            if stale_epochs >= patience:
                break

        total_training_seconds = time.perf_counter() - training_start
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists():
            self.load_checkpoint(best_path)
        self._save_checkpoint("last.pt", len(history))
        collect_diagnostics = bool(
            self.config["evaluation"].get("collect_diagnostics", False)
        )
        test_metrics = self.evaluate(
            "test",
            collect_diagnostics=collect_diagnostics,
            write_predictions=True,
        )
        summary = {
            **test_metrics,
            "dataset": self.config["dataset"]["name"],
            "run_name": self.config["output"].get("run_name", "hra_full"),
            "seed": self.seed,
            "epochs_completed": len(history),
            "training_seconds": total_training_seconds,
            "peak_process_memory_mb": peak_rss / (1024**2),
            "parameters": parameter_count(self.model),
            "device": str(self.device),
        }
        write_json(summary, self.output_dir / "metrics.json")
        if self.writer is not None:
            self.writer.close()
        return summary


def evaluate_checkpoint(
    config: dict[str, Any], checkpoint: str | Path, split: str = "test"
) -> dict[str, Any]:
    trainer = Trainer(config)
    trainer.load_checkpoint(checkpoint)
    metrics = trainer.evaluate(
        split,
        collect_diagnostics=bool(
            config["evaluation"].get("collect_diagnostics", False)
        ),
        write_predictions=True,
    )
    write_json(metrics, trainer.output_dir / f"{split}_metrics.json")
    return metrics
