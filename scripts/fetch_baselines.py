from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def run(*arguments: str, cwd: Path | None = None) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch official baseline repositories at locked commits"
    )
    parser.add_argument(
        "--lock", default="configs/baselines.lock.yaml", help="baseline lock file"
    )
    parser.add_argument("--root", default="external", help="clone destination")
    parser.add_argument("--name", action="append", help="fetch only named baseline")
    arguments = parser.parse_args()

    lock_path = Path(arguments.lock)
    entries = yaml.safe_load(lock_path.read_text(encoding="utf-8"))["baselines"]
    selected = set(arguments.name or entries)
    unknown = selected - set(entries)
    if unknown:
        raise ValueError(f"Unknown baselines: {sorted(unknown)}")
    root = Path(arguments.root)
    root.mkdir(parents=True, exist_ok=True)

    for name, entry in entries.items():
        if name not in selected:
            continue
        destination = root / name
        if not destination.exists():
            run("git", "clone", "--no-checkout", entry["repository"], str(destination))
        run("git", "fetch", "origin", entry["commit"], cwd=destination)
        run("git", "checkout", "--detach", entry["commit"], cwd=destination)
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=destination, text=True
        ).strip()
        if actual != entry["commit"]:
            raise RuntimeError(f"{name}: expected {entry['commit']}, got {actual}")
        print(f"{name}: {actual}")


if __name__ == "__main__":
    main()
