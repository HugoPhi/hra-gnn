from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .data import load_dataset, load_provided_splits, make_splits
from .metrics import anomaly_metrics
from .utils import write_json


def graph_statistics(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset = load_dataset(config["dataset"])
    dataset_config = config["dataset"]
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        graph = dataset[index]
        row: dict[str, Any] = {
            "index": index,
            "graph_id": graph.graph_id,
            "label": graph.label,
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
        }
        for type_id in range(dataset_config["num_node_types"]):
            row[f"node_type_{type_id}"] = int((graph.node_type == type_id).sum())
        for type_id in range(dataset_config["num_edge_types"]):
            row[f"edge_type_{type_id}"] = int((graph.edge_type == type_id).sum())
        rows.append(row)
    frame = pd.DataFrame(rows)
    normal = frame[frame["label"] == 0]
    anomaly = frame[frame["label"] == 1]
    summary = {
        "graphs": len(frame),
        "normal_graphs": len(normal),
        "anomaly_graphs": len(anomaly),
        "anomaly_ratio": len(anomaly) / max(len(frame), 1),
        "mean_nodes": float(frame["num_nodes"].mean()),
        "mean_edges": float(frame["num_edges"].mean()),
        "normal_mean_nodes": float(normal["num_nodes"].mean()),
        "anomaly_mean_nodes": float(anomaly["num_nodes"].mean()),
        "normal_mean_edges": float(normal["num_edges"].mean()),
        "anomaly_mean_edges": float(anomaly["num_edges"].mean()),
        "duplicate_summary_rows": int(
            frame.duplicated(
                subset=[
                    column
                    for column in frame
                    if column.startswith(("num_", "node_type_", "edge_type_"))
                ],
                keep=False,
            ).sum()
        ),
    }
    return frame, summary


def _splits(config: dict[str, Any], dataset) -> dict[str, list[int]]:
    split = config["dataset"].get("split", {})
    if split.get("mode") == "provided":
        return load_provided_splits(
            dataset,
            config["dataset"]["root"],
            train_file=split.get("train_file", "model_gid_list_train.txt"),
            validation_file=split.get("validation_file"),
            test_file=split.get("test_file", "model_gid_list_eval.txt"),
        )
    return make_splits(
        dataset,
        train_normal_ratio=split.get("train_normal_ratio", 0.6),
        validation_normal_ratio=split.get("validation_normal_ratio", 0.2),
        validation_anomaly_ratio=split.get("validation_anomaly_ratio", 0.0),
        seed=config["training"].get("seed", 42),
    )


def simple_feature_baselines(
    config: dict[str, Any], statistics: pd.DataFrame
) -> dict[str, dict[str, float]]:
    dataset = load_dataset(config["dataset"])
    splits = _splits(config, dataset)
    feature_columns = [
        column
        for column in statistics
        if column.startswith(("num_", "node_type_", "edge_type_"))
    ]
    train_x = statistics.loc[splits["train"], feature_columns].to_numpy(float)
    test_x = statistics.loc[splits["test"], feature_columns].to_numpy(float)
    test_y = statistics.loc[splits["test"], "label"].to_numpy(int)
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    test_scaled = scaler.transform(test_x)

    isolation = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=config["training"].get("seed", 42),
    ).fit(train_scaled)
    isolation_score = -isolation.score_samples(test_scaled)

    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(train_scaled)
    ocsvm_score = -ocsvm.score_samples(test_scaled)
    return {
        "IsolationForest": anomaly_metrics(test_y.tolist(), isolation_score.tolist()),
        "OneClassSVM": anomaly_metrics(test_y.tolist(), ocsvm_score.tolist()),
    }


def run_diagnostics(config: dict[str, Any]) -> Path:
    root = (
        Path(config["output"].get("results_root", "artifacts/results"))
        / "diagnostics"
        / config["dataset"]["name"]
    )
    root.mkdir(parents=True, exist_ok=True)
    statistics, summary = graph_statistics(config)
    statistics.to_csv(root / "graph_statistics.csv", index=False)
    summary["simple_baselines"] = simple_feature_baselines(config, statistics)
    write_json(summary, root / "summary.json")
    return root
