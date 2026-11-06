"""
Data preparation script for Laya Anonymization Decision Engine.

Reads the bilingual benchmark datasets (PT and EN), strictly strips data leakage,
performs grouped splitting by dataset_id, and writes out train.jsonl, test.jsonl,
and split_summary.json.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import laya
from src.schema import QUESTIONS, build_column_state


def load_dataset_file(file_path: Path) -> List[Dict[str, Any]]:
    """Loads a benchmark JSON file and returns its datasets list."""
    if not file_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("datasets", [])


def extract_rows_from_datasets(
    datasets: List[Dict[str, Any]],
    lang: str,
    drop_semantic_type: bool = False,
) -> List[Dict[str, Any]]:
    """Flattens dataset columns into row items with sanitized state and targets."""
    rows: List[Dict[str, Any]] = []
    for ds in datasets:
        ds_id = ds.get("dataset_id")
        ds_meta = {
            "dataset_name": ds.get("dataset_name"),
            "domain": ds.get("domain"),
            "purpose": ds.get("purpose"),
            "num_rows": ds.get("num_rows"),
            "num_cols": ds.get("num_cols"),
        }
        for col in ds.get("columns", []):
            target_label = bool(col.get("should_anonymize"))
            prob_true = 1.0 if target_label else 0.0
            prob_false = 0.0 if target_label else 1.0

            state = build_column_state(
                dataset_meta=ds_meta,
                column_data=col,
                drop_semantic_type=drop_semantic_type,
            )

            row = {
                "dataset_id": ds_id,
                "lang": lang,
                "column_name": col.get("name"),
                "state": state,
                "questions": QUESTIONS,
                "expected": {"should_anonymize": target_label},
                "gold": {
                    "should_anonymize": {
                        "label": target_label,
                        "probabilities": {"false": prob_false, "true": prob_true},
                    }
                },
            }
            rows.append(row)
    return rows


def perform_grouped_split(
    all_dataset_ids: List[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """Splits dataset_ids into train and test sets deterministically."""
    sorted_ids = sorted(list(set(all_dataset_ids)))
    rng = random.Random(seed)
    shuffled_ids = list(sorted_ids)
    rng.shuffle(shuffled_ids)

    n_total = len(shuffled_ids)
    n_test = max(1, int(round(n_total * test_size)))
    n_train = n_total - n_test

    train_ids = set(shuffled_ids[:n_train])
    test_ids = set(shuffled_ids[n_train:])
    return train_ids, test_ids


def write_jsonl(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Writes list of dicts to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_split(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates statistics for a set of rows."""
    total = len(rows)
    sensitive = sum(1 for r in rows if r["expected"]["should_anonymize"] is True)
    non_sensitive = total - sensitive
    by_lang: Dict[str, int] = {}
    for r in rows:
        lang = r.get("lang", "unknown")
        by_lang[lang] = by_lang.get(lang, 0) + 1
    return {
        "total": total,
        "sensitive": sensitive,
        "non_sensitive": non_sensitive,
        "sensitive_ratio": round(sensitive / total, 4) if total > 0 else 0.0,
        "by_lang": by_lang,
    }


def prepare_data(
    data_dir: str = "data",
    output_dir: str = "data/processed",
    lang: str = "both",
    test_size: float = 0.2,
    seed: int = 42,
    drop_semantic_type: bool = False,
) -> Dict[str, Any]:
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pt_file = data_path / "anonymization_benchmark_pt.json"
    en_file = data_path / "anonymization_benchmark_en.json"

    datasets_by_lang: Dict[str, List[Dict[str, Any]]] = {}
    if lang in ("pt", "both"):
        datasets_by_lang["pt"] = load_dataset_file(pt_file)
    if lang in ("en", "both"):
        datasets_by_lang["en"] = load_dataset_file(en_file)

    # Collect unique dataset_ids from all loaded datasets
    all_dataset_ids: Set[str] = set()
    for l_key, dsets in datasets_by_lang.items():
        for d in dsets:
            all_dataset_ids.add(d["dataset_id"])

    # Perform grouped split on dataset_ids (identical for PT and EN)
    train_ids, test_ids = perform_grouped_split(
        all_dataset_ids=list(all_dataset_ids),
        test_size=test_size,
        seed=seed,
    )

    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for l_key, dsets in datasets_by_lang.items():
        rows = extract_rows_from_datasets(
            datasets=dsets,
            lang=l_key,
            drop_semantic_type=drop_semantic_type,
        )
        for r in rows:
            if r["dataset_id"] in train_ids:
                train_rows.append(r)
            else:
                test_rows.append(r)

    # Validate that training rows exceed VegaAgent minimum rows requirement
    min_required = laya.VegaAgent.min_rows(QUESTIONS)
    if len(train_rows) < min_required:
        raise ValueError(
            f"Training rows count ({len(train_rows)}) is below VegaAgent.min_rows({min_required})."
        )

    # Shuffle training rows with seed for reproducibility
    rng = random.Random(seed)
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    train_file = out_path / "train.jsonl"
    test_file = out_path / "test.jsonl"
    summary_file = out_path / "split_summary.json"

    write_jsonl(train_rows, train_file)
    write_jsonl(test_rows, test_file)

    train_stats = summarize_split(train_rows)
    test_stats = summarize_split(test_rows)

    summary = {
        "seed": seed,
        "test_size": test_size,
        "lang": lang,
        "drop_semantic_type": drop_semantic_type,
        "min_rows_required": min_required,
        "train_dataset_ids": sorted(list(train_ids)),
        "test_dataset_ids": sorted(list(test_ids)),
        "train_stats": train_stats,
        "test_stats": test_stats,
        "total_rows": len(train_rows) + len(test_rows),
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Data preparation complete!")
    print(f"  Train: {len(train_rows)} rows across {len(train_ids)} datasets saved to {train_file}")
    print(f"  Test:  {len(test_rows)} rows across {len(test_ids)} datasets saved to {test_file}")
    print(f"  Summary saved to {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets for Laya fine-tuning.")
    parser.add_argument("--lang", choices=["pt", "en", "both"], default="both", help="Languages to include.")
    parser.add_argument("--data-dir", default="data", help="Directory containing raw JSON benchmarks.")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for processed JSONL files.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of datasets for test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split.")
    parser.add_argument("--drop-semantic-type", action="store_true", help="Omit semantic_type from column state.")

    args = parser.parse_args()
    prepare_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        lang=args.lang,
        test_size=args.test_size,
        seed=args.seed,
        drop_semantic_type=args.drop_semantic_type,
    )


if __name__ == "__main__":
    main()
