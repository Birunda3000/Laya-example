"""
Predictor wrapper for column and dataset level anonymization decisions.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import laya
from src.schema import QUESTIONS, build_column_state


class AnonymizationDecider:
    """
    High-level decision engine for column and dataset anonymization.
    
    Wraps Laya's VegaAgent, loads fitted logistic heads and conformal
    calibration, and formats responses adhering to governance routing.
    """

    def __init__(
        self,
        calibration_dir: str = "models/vega_anonymizer_calibration",
        use_vega: bool = True,
        default_threshold: float = 0.5,
        alpha: float = 0.10,
        target_precision: float = 0.95,
    ) -> None:
        self.calibration_dir = calibration_dir
        self.use_vega = use_vega
        self.default_threshold = default_threshold
        self.calibration_loaded = False

        if use_vega:
            self.agent = laya.Agent(
                vegaml=True,
                alpha=alpha,
                target_precision=target_precision,
            )
        else:
            self.agent = laya.load("convaiinnovations/laya-multilingual")

        # Load calibration if available
        calib_path = Path(calibration_dir)
        heads_path = calib_path / "heads.npz"
        calib_json = calib_path / "calibration.json"
        if heads_path.exists() and calib_json.exists():
            try:
                self.agent.load_calibration(str(calib_path))
                self.calibration_loaded = True
                print(f"[AnonymizationDecider] Calibration successfully loaded from: {calib_path}")
            except Exception as e:
                print(f"[AnonymizationDecider] Warning: Could not load calibration: {e}")
        else:
            print(f"[AnonymizationDecider] Operating in zero-shot mode (calibration artifacts not found at {calib_path}).")

    def predict_column(
        self,
        dataset_meta: Dict[str, Any],
        column_data: Dict[str, Any],
        threshold: Optional[float] = None,
        drop_semantic_type: bool = False,
    ) -> Dict[str, Any]:
        """Makes an anonymization decision for a single column."""
        if threshold is None:
            threshold = self.default_threshold

        state = build_column_state(
            dataset_meta=dataset_meta,
            column_data=column_data,
            drop_semantic_type=drop_semantic_type,
        )

        out = self.agent.predict(state, QUESTIONS)
        ans = out["answers"]["should_anonymize"]
        score_float = float(ans.get("noul", 0.0))

        return {
            "column_name": column_data.get("name"),
            "score_float": round(score_float, 6),
            "should_anonymize": bool(score_float >= threshold),
            "answer_confidence": float(ans.get("answer_confidence") or 0.0),
            "confidence": float(ans.get("confidence") or 0.0),
            "route": ans.get("route", "auto"),
            "route_reason": ans.get("route_reason", ""),
            "abstain": bool(ans.get("abstain") or False),
            "unbound": bool(ans.get("unbound") or False),
            "prediction_set": ans.get("prediction_set"),
            "latency_ms": out.get("latency_ms", 0.0),
        }

    def predict_dataset(
        self,
        dataset_payload: Dict[str, Any],
        threshold: Optional[float] = None,
        drop_semantic_type: bool = False,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """Makes anonymization decisions for all columns in a dataset."""
        if threshold is None:
            threshold = self.default_threshold

        columns = dataset_payload.get("columns", [])
        if not columns:
            return {
                "dataset_id": dataset_payload.get("dataset_id"),
                "dataset_name": dataset_payload.get("dataset_name"),
                "total_columns": 0,
                "anonymize_columns_count": 0,
                "keep_columns_count": 0,
                "auto_route_count": 0,
                "review_route_count": 0,
                "columns_decisions": [],
            }

        ds_meta = {k: v for k, v in dataset_payload.items() if k != "columns"}
        states = [
            build_column_state(ds_meta, col, drop_semantic_type=drop_semantic_type)
            for col in columns
        ]

        # Use batch prediction for performance
        if hasattr(self.agent, "predict_batch"):
            outputs: List[Dict[str, Any]] = []
            for i in range(0, len(states), batch_size):
                chunk = states[i : i + batch_size]
                outs = self.agent.predict_batch(chunk, questions=QUESTIONS, batch_size=batch_size)
                outputs.extend(outs)
        else:
            outputs = [self.agent.predict(st, QUESTIONS) for st in states]

        columns_decisions: List[Dict[str, Any]] = []
        for col, out in zip(columns, outputs):
            ans = out["answers"]["should_anonymize"]
            score_float = float(ans.get("noul", 0.0))
            decision = {
                "column_name": col.get("name"),
                "score_float": round(score_float, 6),
                "should_anonymize": bool(score_float >= threshold),
                "answer_confidence": float(ans.get("answer_confidence") or 0.0),
                "confidence": float(ans.get("confidence") or 0.0),
                "route": ans.get("route", "auto"),
                "route_reason": ans.get("route_reason", ""),
                "abstain": bool(ans.get("abstain") or False),
                "unbound": bool(ans.get("unbound") or False),
                "prediction_set": ans.get("prediction_set"),
                "latency_ms": out.get("latency_ms", 0.0),
            }
            columns_decisions.append(decision)

        total_cols = len(columns_decisions)
        anonymize_count = sum(1 for d in columns_decisions if d["should_anonymize"])
        keep_count = total_cols - anonymize_count
        auto_count = sum(1 for d in columns_decisions if d["route"] == "auto")
        review_count = total_cols - auto_count

        return {
            "dataset_id": dataset_payload.get("dataset_id"),
            "dataset_name": dataset_payload.get("dataset_name"),
            "domain": dataset_payload.get("domain"),
            "purpose": dataset_payload.get("purpose"),
            "total_columns": total_cols,
            "anonymize_columns_count": anonymize_count,
            "keep_columns_count": keep_count,
            "auto_route_count": auto_count,
            "review_route_count": review_count,
            "columns_decisions": columns_decisions,
        }
