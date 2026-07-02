from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild dataset-isolated TensorBoard curves from history.csv"
    )
    parser.add_argument("--history", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    history = pd.read_csv(arguments.history)
    dataset = arguments.dataset
    output = Path(arguments.output)
    writer = SummaryWriter(output)
    writer.add_custom_scalars(
        {
            dataset: {
                "SVDD Loss: train/test": [
                    "Multiline",
                    [f"{dataset}/Loss/train", f"{dataset}/Loss/test"],
                ],
                "AUC: validation/test": [
                    "Multiline",
                    [f"{dataset}/AUC/validation", f"{dataset}/AUC/test"],
                ],
                "AP: validation/test": [
                    "Multiline",
                    [f"{dataset}/AP/validation", f"{dataset}/AP/test"],
                ],
            }
        }
    )
    for row in history.to_dict(orient="records"):
        step = int(row["epoch"])
        writer.add_scalar(f"{dataset}/Loss/train", row["svdd_loss"], step)
        writer.add_scalar(
            f"{dataset}/Loss/test", row["monitor_test_svdd_loss"], step
        )
        writer.add_scalar(
            f"{dataset}/AUC/validation", row["monitor_validation_auc"], step
        )
        writer.add_scalar(f"{dataset}/AUC/test", row["monitor_test_auc"], step)
        writer.add_scalar(
            f"{dataset}/AP/validation", row["monitor_validation_ap"], step
        )
        writer.add_scalar(f"{dataset}/AP/test", row["monitor_test_ap"], step)
    writer.close()


if __name__ == "__main__":
    main()
