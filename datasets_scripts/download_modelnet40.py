#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target_dir = script_dir / "modelnet_data"
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading ModelNet40 via kagglehub...")
    source_path = Path(kagglehub.dataset_download("balraj98/modelnet40-princeton-3d-object-dataset"))
    print(f"KaggleHub cache path: {source_path}")

    destination = target_dir / source_path.name
    if destination.exists():
        print(f"Removing existing dataset copy at: {destination}")
        shutil.rmtree(destination)

    print(f"Copying dataset into: {destination}")
    shutil.copytree(source_path, destination)

    print(f"Done. Dataset is available under: {destination}")


if __name__ == "__main__":
    main()
