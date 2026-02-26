"""Evaluation helpers for binary classification results."""

from __future__ import annotations

import argparse

import pandas as pd
import sklearn.metrics


def evaluate(
    final_test_results_df: pd.DataFrame,
    label_col: str = "diagonsis",
    pred_col: str | None = None,
    prob_col: str | None = None,
) -> dict[str, float]:
    """Compute and print common binary classification metrics.

    Notes:
    - `label_col` defaults to the original notebook column name `diagonsis`.
    - If `pred_col` / `prob_col` are not provided, the last two columns are used.
    """

    if pred_col is None:
        pred_col = final_test_results_df.columns[-2]
    if prob_col is None:
        prob_col = final_test_results_df.columns[-1]

    label_series = final_test_results_df[label_col]
    preds_series = final_test_results_df[pred_col]
    prob_preds_series = final_test_results_df[prob_col]

    metrics = {
        "f1_micro": float(sklearn.metrics.f1_score(label_series, preds_series, average="micro")),
        "balanced_accuracy": float(
            sklearn.metrics.balanced_accuracy_score(label_series, preds_series),
        ),
        "auroc": float(sklearn.metrics.roc_auc_score(label_series, prob_preds_series)),
        "auprc": float(
            sklearn.metrics.average_precision_score(label_series, prob_preds_series, pos_label=1),
        ),
        "specificity": float(
            ((label_series == 0) & (preds_series == 0)).sum() / (label_series == 0).sum(),
        ),
        "sensitivity": float(
            ((label_series == 1) & (preds_series == 1)).sum() / (label_series == 1).sum(),
        ),
    }

    print(f"F1: {metrics['f1_micro']:.2f}")
    print(f"BA: {metrics['balanced_accuracy']:.2f}")
    print(f"AUROC: {metrics['auroc']:.2f}")
    print(f"AUPRC: {metrics['auprc']:.2f}")
    print(f"Specificity: {metrics['specificity']:.2f}")
    print(f"Sensitivity: {metrics['sensitivity']:.2f}")
    print()

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CSV file of predictions.")
    parser.add_argument("--results-csv", required=True, help="Path to a results CSV file.")
    parser.add_argument("--label-col", default="diagonsis", help="Label column name.")
    parser.add_argument(
        "--pred-col",
        default=None,
        help="Prediction label column name (default: second-to-last column).",
    )
    parser.add_argument(
        "--prob-col",
        default=None,
        help="Prediction probability column name (default: last column).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results_df = pd.read_csv(args.results_csv)
    evaluate(results_df, label_col=args.label_col, pred_col=args.pred_col, prob_col=args.prob_col)


if __name__ == "__main__":
    main()
