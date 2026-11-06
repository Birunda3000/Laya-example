"""
Zero-Shot Evaluation, CPU Training via VegaAgent.fit(), and Post-Fit Evaluation.

Performs:
  1. Zero-shot baseline evaluation on test split
  2. agent.fit() over frozen universal reader features (logistic heads, temperature, conformal q-hat, precision band)
  3. Saves calibration artifacts (heads.npz + calibration.json)
  4. Post-fit evaluation and Before/After comparison
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import laya
from src.schema import QUESTIONS


def compute_roc_auc(scores: List[float], labels: List[bool]) -> float:
    """Computes exact ROC-AUC using Wilcoxon-Mann-Whitney rank sum."""
    pos_scores = [s for s, y in zip(scores, labels) if y]
    neg_scores = [s for s, y in zip(scores, labels) if not y]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Pair with indices and sort
    paired = sorted([(s, y) for s, y in zip(scores, labels)], key=lambda x: x[0])
    
    # Calculate ranks with ties handled by average rank
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j < len(paired) and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    sum_pos_ranks = sum(r for r, (_, y) in zip(ranks, paired) if y)
    u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def evaluate_model(
    agent: Any,
    rows: List[Dict[str, Any]],
    questions: Dict[str, Any] = QUESTIONS,
    batch_size: int = 16,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluates agent on test rows, computing classification, calibration, and routing metrics."""
    states = [r["state"] for r in rows]
    gold_labels = [bool(r["expected"]["should_anonymize"]) for r in rows]

    # Predict in batches
    outputs: List[Dict[str, Any]] = []
    for i in range(0, len(states), batch_size):
        chunk = states[i : i + batch_size]
        outs = agent.predict_batch(chunk, questions=questions, batch_size=batch_size)
        outputs.extend(outs)

    scores: List[float] = []
    preds: List[bool] = []
    routes: List[str] = []
    abstains: List[bool] = []
    unbounds: List[bool] = []

    for out in outputs:
        ans = out["answers"]["should_anonymize"]
        score = float(ans.get("noul", 0.0))
        scores.append(score)
        preds.append(score >= threshold)
        routes.append(ans.get("route", "auto"))
        abstains.append(bool(ans.get("abstain", False)))
        unbounds.append(bool(ans.get("unbound", False)))

    n_total = len(gold_labels)
    tp = sum(1 for p, y in zip(preds, gold_labels) if p and y)
    fp = sum(1 for p, y in zip(preds, gold_labels) if p and not y)
    tn = sum(1 for p, y in zip(preds, gold_labels) if not p and not y)
    fn = sum(1 for p, y in zip(preds, gold_labels) if not p and y)

    acc = (tp + tn) / n_total if n_total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    brier = sum((s - (1.0 if y else 0.0)) ** 2 for s, y in zip(scores, gold_labels)) / n_total
    roc_auc = compute_roc_auc(scores, gold_labels)

    # Routing metrics
    auto_indices = [i for i, r in enumerate(routes) if r == "auto"]
    review_indices = [i for i, r in enumerate(routes) if r == "review"]
    n_auto = len(auto_indices)
    n_review = len(review_indices)

    auto_tp = sum(1 for i in auto_indices if preds[i] and gold_labels[i])
    auto_fp = sum(1 for i in auto_indices if preds[i] and not gold_labels[i])
    auto_correct = sum(1 for i in auto_indices if preds[i] == gold_labels[i])
    auto_prec = auto_tp / (auto_tp + auto_fp) if (auto_tp + auto_fp) > 0 else 0.0
    auto_acc = auto_correct / n_auto if n_auto > 0 else 0.0

    return {
        "total": n_total,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "brier_score": round(brier, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "routing": {
            "auto_count": n_auto,
            "review_count": n_review,
            "auto_rate": round(n_auto / n_total, 4) if n_total > 0 else 0.0,
            "review_rate": round(n_review / n_total, 4) if n_total > 0 else 0.0,
            "auto_precision": round(auto_prec, 4),
            "auto_accuracy": round(auto_acc, 4),
            "abstain_count": sum(abstains),
            "unbound_count": sum(unbounds),
        },
    }


def print_metrics_summary(title: str, m: Dict[str, Any]) -> None:
    print(f"\n==================== {title} ====================")
    print(f"Total Rows:     {m['total']}")
    print(f"Accuracy:       {m['accuracy']:.4f}")
    print(f"Precision:      {m['precision']:.4f}")
    print(f"Recall:         {m['recall']:.4f}")
    print(f"F1-Score:       {m['f1']:.4f}")
    print(f"Brier Score:    {m['brier_score']:.4f}  (lower is better)")
    print(f"ROC-AUC:        {m['roc_auc']:.4f}")
    r = m["routing"]
    print(f"Route 'auto':   {r['auto_count']} ({r['auto_rate']*100:.1f}%) | Auto Precision: {r['auto_precision']:.4f} | Auto Acc: {r['auto_accuracy']:.4f}")
    print(f"Route 'review': {r['review_count']} ({r['review_rate']*100:.1f}%) | Abstains: {r['abstain_count']} | Unbound: {r['unbound_count']}")
    print("=" * (len(title) + 42))


def print_comparison_table(before: Optional[Dict[str, Any]], after: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print(f"{'METRIC':<25} | {'BEFORE FIT (ZERO-SHOT)':<20} | {'AFTER FIT':<15}")
    print("-" * 70)
    metrics_list = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1"),
        ("Brier Score", "brier_score"),
        ("ROC-AUC", "roc_auc"),
    ]
    for label, key in metrics_list:
        b_val = f"{before[key]:.4f}" if before else "N/A (skipped)"
        a_val = f"{after[key]:.4f}"
        print(f"{label:<25} | {b_val:<20} | {a_val:<15}")
    print("-" * 70)
    b_auto = f"{before['routing']['auto_rate']*100:.1f}%" if before else "N/A"
    a_auto = f"{after['routing']['auto_rate']*100:.1f}%"
    print(f"{'Auto Route Rate':<25} | {b_auto:<20} | {a_auto:<15}")
    b_ap = f"{before['routing']['auto_precision']:.4f}" if before else "N/A"
    a_ap = f"{after['routing']['auto_precision']:.4f}"
    print(f"{'Auto Precision':<25} | {b_ap:<20} | {a_ap:<15}")
    print("=" * 70)


def train_and_calibrate(
    train_file: str = "data/processed/train.jsonl",
    test_file: str = "data/processed/test.jsonl",
    calibration_dir: str = "models/vega_anonymizer_calibration",
    target_precision: float = 0.95,
    alpha: float = 0.10,
    folds: int = 5,
    batch_size: int = 16,
    skip_zero_shot: bool = False,
) -> Dict[str, Any]:
    train_path = Path(train_file)
    test_path = Path(test_file)
    calib_path = Path(calibration_dir)
    calib_path.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    # Load test rows
    with open(test_path, "r", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]

    # Load train rows
    with open(train_path, "r", encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f]

    print(f"Loaded {len(train_rows)} train rows and {len(test_rows)} test rows.")
    print("Initializing VegaAgent (universal reader nandakishorm/vega-U)...")
    agent = laya.Agent(
        vegaml=True,
        alpha=alpha,
        target_precision=target_precision,
        folds=folds,
    )

    before_metrics = None
    if not skip_zero_shot:
        print("\n[STEP 1/3] Running Zero-Shot Evaluation on held-out test rows...")
        before_metrics = evaluate_model(agent, test_rows, QUESTIONS, batch_size=batch_size)
        print_metrics_summary("ZERO-SHOT EVALUATION (BEFORE FIT)", before_metrics)
    else:
        print("\n[STEP 1/3] Skipping Zero-Shot Evaluation as requested.")

    print(f"\n[STEP 2/3] Fitting VegaAgent on {len(train_rows)} rows (folds={folds}, target_precision={target_precision})...")
    fit_report = agent.fit(
        train_rows,
        questions=QUESTIONS,
        folds=folds,
        calibrate=True,
        target_precision=target_precision,
    )

    print("\n--- Fit Report ---")
    print(json.dumps(fit_report, indent=2, default=str))

    # Save calibration artifacts
    print(f"\nSaving calibration to: {calib_path}...")
    save_result = agent.save_calibration(str(calib_path))
    print(f"Calibration saved: {save_result}")

    # Verify artifacts
    heads_file = calib_path / "heads.npz"
    calib_file = calib_path / "calibration.json"
    if heads_file.exists() and calib_file.exists():
        print(f" Verified artifacts: {heads_file} ({heads_file.stat().st_size} bytes), {calib_file} ({calib_file.stat().st_size} bytes)")
    else:
        print(f"⚠️ Warning: Expected artifact files not both found in {calib_path}")

    print("\n[STEP 3/3] Running Post-Fit Evaluation on held-out test rows...")
    after_metrics = evaluate_model(agent, test_rows, QUESTIONS, batch_size=batch_size)
    print_metrics_summary("POST-FIT EVALUATION (AFTER FIT)", after_metrics)

    print_comparison_table(before_metrics, after_metrics)

    evaluation_report = {
        "train_rows_count": len(train_rows),
        "test_rows_count": len(test_rows),
        "folds": folds,
        "target_precision": target_precision,
        "alpha": alpha,
        "fit_report": fit_report,
        "before_fit": before_metrics,
        "after_fit": after_metrics,
    }

    report_file = calib_path / "evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2, default=str)
    print(f"\nFull evaluation report saved to {report_file}")

    return evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Laya VegaAgent on anonymization benchmark.")
    parser.add_argument("--train-file", default="data/processed/train.jsonl", help="Path to train jsonl.")
    parser.add_argument("--test-file", default="data/processed/test.jsonl", help="Path to test jsonl.")
    parser.add_argument("--calibration-dir", default="models/vega_anonymizer_calibration", help="Directory to save heads & calibration.")
    parser.add_argument("--target-precision", type=float, default=0.95, help="Target precision for review band.")
    parser.add_argument("--alpha", type=float, default=0.10, help="Conformal alpha.")
    parser.add_argument("--folds", type=int, default=5, help="Cross-validation folds.")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--skip-zero-shot", action="store_true", help="Skip pre-fit evaluation.")

    args = parser.parse_args()
    train_and_calibrate(
        train_file=args.train_file,
        test_file=args.test_file,
        calibration_dir=args.calibration_dir,
        target_precision=args.target_precision,
        alpha=args.alpha,
        folds=args.folds,
        batch_size=args.batch_size,
        skip_zero_shot=args.skip_zero_shot,
    )


if __name__ == "__main__":
    main()
