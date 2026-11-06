"""
Unit tests for src/schema.py: questions schema and data leakage prevention.
"""

import unittest
from src.schema import QUESTIONS, build_column_state


class TestSchemaAndLeakage(unittest.TestCase):

    def test_questions_structure(self):
        """Validates that QUESTIONS follows the exact noul specification."""
        self.assertIn("should_anonymize", QUESTIONS)
        q = QUESTIONS["should_anonymize"]
        self.assertEqual(q["type"], "noul")
        self.assertIn("instructions", q)
        self.assertIn("criteria", q)
        self.assertIn("true", q["criteria"])
        self.assertIn("false", q["criteria"])
        self.assertEqual(q["labels"], {"true": "A", "false": "B"})

    def test_build_column_state_strips_leakage(self):
        """Ensures should_anonymize, reason, and dataset_id are never leaked into state."""
        dataset_meta = {
            "dataset_id": "secret_dataset_123",
            "dataset_name": "Patients",
            "domain": "Healthcare",
            "purpose": "Clinical analysis",
            "num_rows": 1000,
            "num_cols": 5,
        }
        column_data = {
            "name": "patient_ssn",
            "description": "Social security number of patient",
            "data_type": "string",
            "semantic_type": "ssn",
            "null_percentage": 0.0,
            "cardinality": 950,
            "quantiles": None,
            "detectors": {"ssn": True, "cpf": False},
            # Leakage fields:
            "should_anonymize": True,
            "reason": "Direct PII attribute identifying patient.",
            "dataset_id": "secret_dataset_123",
        }

        state = build_column_state(dataset_meta, column_data, drop_semantic_type=False)

        # Check dataset section
        self.assertIn("dataset", state)
        self.assertNotIn("dataset_id", state["dataset"])
        self.assertEqual(state["dataset"]["dataset_name"], "Patients")

        # Check column section
        self.assertIn("column", state)
        col = state["column"]
        self.assertNotIn("should_anonymize", col)
        self.assertNotIn("reason", col)
        self.assertNotIn("dataset_id", col)
        self.assertEqual(col["name"], "patient_ssn")
        self.assertEqual(col["semantic_type"], "ssn")
        self.assertIsNone(col["quantiles"])

    def test_build_column_state_drop_semantic_type(self):
        """Ensures semantic_type is excluded when drop_semantic_type=True."""
        dataset_meta = {"dataset_name": "Test", "domain": "General"}
        column_data = {
            "name": "col_a",
            "data_type": "string",
            "semantic_type": "email",
        }
        state = build_column_state(dataset_meta, column_data, drop_semantic_type=True)
        self.assertNotIn("semantic_type", state["column"])


if __name__ == "__main__":
    unittest.main()
