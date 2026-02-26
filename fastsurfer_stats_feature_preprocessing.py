"""Extract FastSurfer morphology features from `.stats` files.

This script reads a CSV file containing subject/session identifiers and writes
one `.npy` feature file per row.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_EXPECTED_LENGTH = 700


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract morphology features from FastSurfer aseg+DKT stats files.",
    )
    parser.add_argument(
        "--dataset-csv",
        required=True,
        help="CSV path containing subject/session columns.",
    )
    parser.add_argument(
        "--stats-path-template",
        required=True,
        help=(
            "Stats file template. Use {subject_id} and {session_id} placeholders, "
            "for example: /data/fastsurfer/{subject_id}_{session_id}/stats/aseg+DKT.stats"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save `mor_features_<subject>_<session>.npy` files.",
    )
    parser.add_argument(
        "--subject-col",
        default=None,
        help="Subject ID column name. Defaults to the first CSV column.",
    )
    parser.add_argument(
        "--session-col",
        default=None,
        help="Session ID column name. Defaults to the second CSV column.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of rows to process.",
    )
    parser.add_argument(
        "--expected-length",
        type=int,
        default=DEFAULT_EXPECTED_LENGTH,
        help="Expected flattened feature length for sanity checking.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on the first missing/invalid stats file instead of skipping.",
    )
    return parser.parse_args()


def mor_feature_from_stats(aseg_stats_path: Path) -> np.ndarray:
    """Convert one `aseg+DKT.stats` file into a flattened feature array."""
    with aseg_stats_path.open("r", encoding="utf-8") as stats_file:
        aseg_stats_txt = stats_file.readlines()

    mor_feature_rows = [line.split() for line in aseg_stats_txt[54:]]
    col_names = mor_feature_rows[0][2:]
    mor_feature_df = pd.DataFrame(mor_feature_rows[1:], columns=col_names).drop(
        columns=["Index", "SegId", "StructName"],
    )

    return mor_feature_df.to_numpy().flatten()


def resolve_columns(df: pd.DataFrame, subject_col: str | None, session_col: str | None) -> tuple[str, str]:
    """Resolve subject/session column names from CLI args or defaults."""
    if subject_col is None:
        subject_col = df.columns[0]
    if session_col is None:
        if len(df.columns) < 2:
            raise ValueError("CSV must have at least two columns or provide --session-col.")
        session_col = df.columns[1]

    missing_cols = [col for col in (subject_col, session_col) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    return subject_col, session_col


def main() -> None:
    """Run feature extraction for all rows in the input CSV."""
    args = parse_args()

    dataset_csv = Path(args.dataset_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_csv)
    subject_col, session_col = resolve_columns(df, args.subject_col, args.session_col)

    total_rows = len(df) if args.max_rows is None else min(len(df), args.max_rows)
    failures: list[tuple[int, str, str]] = []

    for idx in tqdm(range(total_rows), desc="Extracting morphology features"):
        subject_id = str(df.loc[idx, subject_col])
        session_id = str(df.loc[idx, session_col])
        stats_path = Path(
            args.stats_path_template.format(subject_id=subject_id, session_id=session_id),
        )

        try:
            mor_features_array = mor_feature_from_stats(stats_path)
        except Exception as exc:  # noqa: BLE001
            message = f"row={idx}, subject={subject_id}, session={session_id}, error={exc}"
            if args.strict:
                raise RuntimeError(message) from exc
            print(f"[WARN] Skip: {message}")
            failures.append((idx, subject_id, session_id))
            continue

        if args.expected_length and len(mor_features_array) != args.expected_length:
            print(
                "[WARN] Unexpected feature length: "
                f"row={idx}, subject={subject_id}, session={session_id}, "
                f"expected={args.expected_length}, got={len(mor_features_array)}",
            )

        output_file = output_dir / f"mor_features_{subject_id}_{session_id}.npy"
        with output_file.open("wb") as f:
            np.save(f, mor_features_array)

    print(f"Done. Processed {total_rows} rows. Failed: {len(failures)}")


if __name__ == "__main__":
    main()
