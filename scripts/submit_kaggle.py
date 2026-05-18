#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_COMPETITION = "data-mining-2026-final-project"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and optionally submit a CSV to Kaggle.")
    parser.add_argument("submission", help="Submission CSV path.")
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--message", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only validate and print the Kaggle command.")
    return parser.parse_args()


def run_validation(submission: Path, sample_submission: Path) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("validate_submission.py")),
        str(submission),
        "--sample-submission",
        str(sample_submission),
    ]
    subprocess.run(command, check=True)


def find_kaggle_executable() -> str | None:
    executable = shutil.which("kaggle")
    if executable is not None:
        return executable
    python_bin = Path(sys.executable)
    local_kaggle = python_bin.with_name("kaggle")
    if local_kaggle.exists():
        return str(local_kaggle)
    return None


def main() -> None:
    args = parse_args()
    submission = Path(args.submission)
    sample_submission = Path(args.sample_submission)
    if not submission.exists():
        raise FileNotFoundError(f"Submission not found: {submission}")
    if not sample_submission.exists():
        raise FileNotFoundError(f"Sample submission not found: {sample_submission}")

    run_validation(submission, sample_submission)

    message = args.message or f"Validated submission {submission.name}"
    kaggle_executable = find_kaggle_executable()
    kaggle_command = [
        kaggle_executable or "kaggle",
        "competitions",
        "submit",
        "-c",
        args.competition,
        "-f",
        str(submission),
        "-m",
        message,
    ]
    print("Kaggle command:")
    print(" ".join(kaggle_command))

    if args.dry_run:
        print("Dry run only; not submitting.")
        return

    if kaggle_executable is None:
        raise RuntimeError(
            "Kaggle CLI is not installed. Install/configure it or upload the validated CSV manually on Kaggle."
        )
    env = os.environ.copy()
    local_config = Path(".kaggle")
    if "KAGGLE_CONFIG_DIR" not in env and (local_config / "kaggle.json").exists():
        env["KAGGLE_CONFIG_DIR"] = str(local_config)
    local_access_token = local_config / "access_token"
    if "KAGGLE_API_TOKEN" not in env and local_access_token.exists():
        env["KAGGLE_API_TOKEN"] = local_access_token.read_text(encoding="utf-8").strip()
    subprocess.run(kaggle_command, check=True, env=env)


if __name__ == "__main__":
    main()
