#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a project-local Kaggle config from environment variables.")
    parser.add_argument("--output-dir", default=".kaggle")
    parser.add_argument(
        "--token-env",
        default=None,
        help="Name of an environment variable containing kaggle.json JSON or base64-encoded JSON.",
    )
    parser.add_argument("--username", default=None, help="Literal Kaggle username.")
    parser.add_argument("--username-env", default=None, help="Name of an environment variable containing username.")
    parser.add_argument("--key-env", default=None, help="Name of an environment variable containing key.")
    return parser.parse_args()


def parse_token(value: str) -> dict[str, str] | None:
    candidates = [value]
    try:
        candidates.append(base64.b64decode(value).decode("utf-8"))
    except Exception:
        pass
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        username = parsed.get("username")
        key = parsed.get("key")
        if username and key:
            return {"username": str(username), "key": str(key)}
    return None


def load_credentials(
    token_env: str | None = None,
    username_literal: str | None = None,
    username_env: str | None = None,
    key_env: str | None = None,
) -> dict[str, str]:
    username = username_literal or os.environ.get(username_env or "KAGGLE_USERNAME")
    key = os.environ.get(key_env or "KAGGLE_KEY")
    if username and key:
        return {"username": username, "key": key}

    token_names = [name for name in (token_env, "KAGGLE_API_TOKEN", "KAGGLE_API_TOKE", "KGAT") if name]
    for name in token_names:
        value = os.environ.get(name)
        if not value:
            continue
        parsed = parse_token(value)
        if parsed:
            return parsed
        raise ValueError(
            f"{name} is set, but it is not JSON/base64 JSON with username and key fields."
        )

    raise ValueError(
        "No Kaggle credentials found. Set KAGGLE_USERNAME and KAGGLE_KEY, "
        "pass --username with --key-env, or set KAGGLE_API_TOKEN/KAGGLE_API_TOKE "
        "to the kaggle.json content."
    )


def main() -> None:
    args = parse_args()
    credentials = load_credentials(args.token_env, args.username, args.username_env, args.key_env)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "kaggle.json"
    output_path.write_text(json.dumps(credentials, indent=2) + "\n", encoding="utf-8")
    output_path.chmod(0o600)
    print(f"Wrote {output_path}")
    print("Use this config with: KAGGLE_CONFIG_DIR=.kaggle .venv/bin/kaggle ...")


if __name__ == "__main__":
    main()
