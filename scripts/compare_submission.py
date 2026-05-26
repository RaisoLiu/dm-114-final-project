#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two Kaggle submission CSV files.")
    parser.add_argument("generated", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--abs-tol", type=float, default=1e-9)
    parser.add_argument("--ulp-tol", type=int, default=16)
    return parser.parse_args()


def ordered_float_bits(value: float) -> int:
    bits = struct.unpack("!q", struct.pack("!d", value))[0]
    return bits if bits >= 0 else 0x8000000000000000 - bits


def ulp_diff(left: float, right: float) -> int:
    return abs(ordered_float_bits(left) - ordered_float_bits(right))


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        return list(reader.fieldnames), list(reader)


def maybe_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        raise ValueError(f"Non-finite value in submission: {value}")
    return parsed


def main() -> None:
    args = parse_args()
    gen_header, gen_rows = read_rows(args.generated)
    ref_header, ref_rows = read_rows(args.reference)

    if gen_header != ref_header:
        raise SystemExit(f"Header mismatch:\n generated={gen_header}\n reference={ref_header}")
    if len(gen_rows) != len(ref_rows):
        raise SystemExit(f"Row-count mismatch: generated={len(gen_rows)} reference={len(ref_rows)}")

    max_abs = 0.0
    max_ulp = 0
    max_cell = None
    exact_string_mismatches = 0

    for row_idx, (gen_row, ref_row) in enumerate(zip(gen_rows, ref_rows), start=2):
        for column in gen_header:
            gen_value = gen_row[column]
            ref_value = ref_row[column]
            gen_float = maybe_float(gen_value)
            ref_float = maybe_float(ref_value)
            if gen_float is None or ref_float is None:
                if gen_value != ref_value:
                    exact_string_mismatches += 1
                    max_cell = (row_idx, column, gen_value, ref_value)
                continue
            abs_diff = abs(gen_float - ref_float)
            cell_ulp = ulp_diff(gen_float, ref_float)
            if abs_diff > max_abs or cell_ulp > max_ulp:
                max_abs = max(max_abs, abs_diff)
                max_ulp = max(max_ulp, cell_ulp)
                max_cell = (row_idx, column, gen_value, ref_value)

    print(f"Compared {len(gen_rows)} rows and {len(gen_header)} columns")
    print(f"max_abs_diff={max_abs:.12g}")
    print(f"max_ulp_diff={max_ulp}")
    if max_cell is not None:
        row_idx, column, gen_value, ref_value = max_cell
        print(f"max_cell=row {row_idx}, column {column}: generated={gen_value}, reference={ref_value}")

    if exact_string_mismatches:
        raise SystemExit(f"{exact_string_mismatches} non-numeric cells differ")
    if max_abs > args.abs_tol and max_ulp > args.ulp_tol:
        raise SystemExit(
            f"Numeric diff exceeds tolerances: max_abs={max_abs} > {args.abs_tol} "
            f"and max_ulp={max_ulp} > {args.ulp_tol}"
        )
    print("Submission comparison passed")


if __name__ == "__main__":
    main()
