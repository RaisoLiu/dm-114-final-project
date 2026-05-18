#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PRED_COLS = [f"pred_week{i}" for i in range(1, 6)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the 2026-05-10 08:10 submission manifest.")
    parser.add_argument("--manifest", default="reports/submit_manifest_20260510_0810.json")
    parser.add_argument("--submit-script", default="scripts/submit_three_20260510_0810.sh")
    parser.add_argument("--atol", type=float, default=1e-9)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bash_array(script_text: str, name: str) -> list[str]:
    match = re.search(rf"{name}=\(\n(?P<body>.*?)\n\)", script_text, flags=re.S)
    if not match:
        raise AssertionError(f"Could not find bash array {name}.")
    values: list[str] = []
    for line in match.group("body").splitlines():
        line = line.strip()
        if not line:
            continue
        item_match = re.fullmatch(r'"([^"]+)"', line)
        if not item_match:
            raise AssertionError(f"Could not parse {name} item: {line}")
        values.append(item_match.group(1))
    return values


def check_close(label: str, actual: float, expected: float, atol: float) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{label}: actual {actual!r} != expected {expected!r}")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    submit_script_path = Path(args.submit_script)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    queue = manifest["queue"]

    script_text = submit_script_path.read_text(encoding="utf-8")
    script_files = parse_bash_array(script_text, "FILES")
    script_messages = parse_bash_array(script_text, "MESSAGES")
    manifest_files = [item["path"] for item in queue]
    manifest_messages = [item["message"] for item in queue]
    if script_files != manifest_files:
        raise AssertionError(f"Submit script FILES differ from manifest: {script_files} != {manifest_files}")
    if script_messages != manifest_messages:
        raise AssertionError("Submit script MESSAGES differ from manifest.")

    for expected_order, item in enumerate(queue, start=1):
        if item["order"] != expected_order:
            raise AssertionError(f"Queue order mismatch at item {expected_order}: {item['order']}")
        path = Path(item["path"])
        if not path.exists():
            raise AssertionError(f"Missing submission file: {path}")
        actual_sha = sha256(path)
        if actual_sha != item["sha256"]:
            raise AssertionError(f"{path}: sha256 {actual_sha} != {item['sha256']}")

        df = pd.read_csv(path)
        missing_cols = [col for col in ["region_id", *PRED_COLS] if col not in df.columns]
        if missing_cols:
            raise AssertionError(f"{path}: missing columns {missing_cols}")
        if len(df) != item["rows"]:
            raise AssertionError(f"{path}: rows {len(df)} != {item['rows']}")
        predictions = df[PRED_COLS].to_numpy(dtype=float)
        if not np.isfinite(predictions).all():
            raise AssertionError(f"{path}: predictions contain non-finite values")

        check_close(f"{path} prediction_min", float(predictions.min()), item["prediction_min"], args.atol)
        check_close(f"{path} prediction_max", float(predictions.max()), item["prediction_max"], args.atol)
        check_close(f"{path} prediction_mean", float(predictions.mean()), item["prediction_mean"], args.atol)
        check_close(f"{path} prediction_std", float(predictions.std()), item["prediction_std"], args.atol)
        print(f"verified {expected_order}: {path}  sha256={actual_sha}")

    print(f"Manifest verified: {manifest_path}")


if __name__ == "__main__":
    main()
