#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile


DEFAULT_COMPETITION = "data-mining-2026-final-project"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract Kaggle competition data.")
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--kaggle-config-dir", default=".kaggle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = args.kaggle_config_dir

    command = [
        ".venv/bin/kaggle",
        "competitions",
        "download",
        "-c",
        args.competition,
        "-p",
        str(data_dir),
    ]
    subprocess.run(command, check=True, env=env)

    for zip_path in data_dir.glob("*.zip"):
        with ZipFile(zip_path) as archive:
            archive.extractall(data_dir)
        print(f"Extracted {zip_path}")


if __name__ == "__main__":
    main()

