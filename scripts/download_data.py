from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import gdown


DATASETS = {
    "flowgraph": {
        "id": "1vDuDe6c76cYz6x2yKaeO2gpsGc7b7yiw",
        "directory": "ProcessedData_FlowGraph",
    },
    "tracelog": {
        "id": "1IH_GwrbMNl1gm8O6uuTR5qprhdhkISvz",
        "directory": "ProcessedData_TraceLog",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=(*DATASETS, "all"))
    parser.add_argument("--root", default="data")
    parser.add_argument("--keep-archive", action="store_true")
    arguments = parser.parse_args()

    names = DATASETS if arguments.dataset == "all" else [arguments.dataset]
    root = Path(arguments.root)
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        metadata = DATASETS[name]
        archive = root / f"{name}.zip"
        destination = root / metadata["directory"]
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name}...")
        gdown.download(
            id=metadata["id"],
            output=str(archive),
            quiet=False,
            resume=True,
        )
        print(f"Extracting {archive} to {destination}...")
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)
        if not arguments.keep_archive:
            archive.unlink()
        print(f"{name} ready at {destination}")


if __name__ == "__main__":
    main()
