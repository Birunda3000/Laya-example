"""
RLCD / Weight fine-tuning module for Laya Anonymization Decision Engine.

Supports dry-run validation of dataset tokenization and configuration,
as well as full RLCD fine-tuning using laya.train.finetune.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

from laya.train import TrainConfig, dry_run, finetune


def run_training_pipeline(
    data: str = "data/processed/train.jsonl",
    eval_data: Optional[str] = "data/processed/test.jsonl",
    base: str = "convaiinnovations/laya-multilingual",
    output_dir: str = "models/laya_rlcd_checkpoint",
    is_dry_run: bool = False,
    epochs: int = 4,
    micro_batch: int = 1,
    grad_accum: int = 16,
    encoder_lr: float = 2.5e-5,
    head_lr: float = 1e-4,
    loss: str = "rlcd",
    device: str = "auto",
) -> Dict[str, Any]:
    if not os.path.exists(data):
        raise FileNotFoundError(f"Training data not found at: {data}")
    if eval_data and not os.path.exists(eval_data):
        print(f"Warning: Evaluation data {eval_data} does not exist; setting to None.")
        eval_data = None

    config = TrainConfig(
        epochs=epochs,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        loss=loss,
        eval_data=eval_data,
    )

    if is_dry_run:
        print(f"[DRY-RUN] Validating data tokenization and structure with base model: {base}...")
        report = dry_run(data=data, model_dir=base, config=config)
        print("\n--- Dry Run Report ---")
        print(json.dumps(report, indent=2, default=str))
        return report
    else:
        print(f"[FINETUNE] Starting fine-tuning from {base} to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        report = finetune(
            data=data,
            model_dir=base,
            output_dir=output_dir,
            config=config,
            device=device,
        )
        print("\n--- Fine-tuning Report ---")
        print(json.dumps(report, indent=2, default=str))
        return report


def main():
    parser = argparse.ArgumentParser(description="RLCD fine-tuning and dry-run for Laya Anonymization Decider.")
    parser.add_argument("--data", default="data/processed/train.jsonl", help="Path to training jsonl.")
    parser.add_argument("--eval", default="data/processed/test.jsonl", help="Path to eval jsonl.")
    parser.add_argument("--base", default="convaiinnovations/laya-multilingual", help="Base model identifier or path.")
    parser.add_argument("--output-dir", default="models/laya_rlcd_checkpoint", help="Directory to save checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Run tokenization dry-run without training.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs.")
    parser.add_argument("--micro-batch", type=int, default=1, help="Micro-batch size.")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument("--encoder-lr", type=float, default=2.5e-5, help="Learning rate for encoder.")
    parser.add_argument("--head-lr", type=float, default=1e-4, help="Learning rate for decision head.")
    parser.add_argument("--loss", choices=["rlcd", "soft_ce"], default="rlcd", help="Loss function.")
    parser.add_argument("--device", default="auto", help="Compute device ('auto', 'cpu', 'cuda').")

    args = parser.parse_args()
    run_training_pipeline(
        data=args.data,
        eval_data=args.eval,
        base=args.base,
        output_dir=args.output_dir,
        is_dry_run=args.dry_run,
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        loss=args.loss,
        device=args.device,
    )


if __name__ == "__main__":
    main()
