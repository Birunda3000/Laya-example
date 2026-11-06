"""
Unit tests for data split integrity and leakage verification.
"""

import json
import unittest
from pathlib import Path


class TestDataIntegrity(unittest.TestCase):

    def setUp(self):
        self.train_path = Path("data/processed/train.jsonl")
        self.test_path = Path("data/processed/test.jsonl")
        self.summary_path = Path("data/processed/split_summary.json")

    def test_files_exist(self):
        """Verifies that processed files are present."""
        self.assertTrue(self.train_path.exists(), "train.jsonl does not exist")
        self.assertTrue(self.test_path.exists(), "test.jsonl does not exist")
        self.assertTrue(self.summary_path.exists(), "split_summary.json does not exist")

    def test_zero_leakage_between_splits(self):
        """Verifies that no dataset_id is present in both train and test splits."""
        train_ids = set()
        with open(self.train_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                train_ids.add(row["dataset_id"])

        test_ids = set()
        with open(self.test_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                test_ids.add(row["dataset_id"])

        intersection = train_ids.intersection(test_ids)
        self.assertEqual(len(intersection), 0, f"Data leakage detected! Overlapping dataset_ids: {intersection}")
        self.assertEqual(len(train_ids), 16)
        self.assertEqual(len(test_ids), 4)

    def test_row_schema_compatibility(self):
        """Verifies that each row has state, expected, and gold formatted for Laya & VegaML."""
        with open(self.train_path, "r", encoding="utf-8") as f:
            sample_line = f.readline()
            row = json.loads(sample_line)

        self.assertIn("dataset_id", row)
        self.assertIn("lang", row)
        self.assertIn("column_name", row)
        self.assertIn("state", row)
        self.assertIn("expected", row)
        self.assertIn("gold", row)

        # Check state does not leak target
        self.assertNotIn("should_anonymize", row["state"]["column"])
        self.assertNotIn("reason", row["state"]["column"])

        # Check expected & gold formatting
        self.assertIsInstance(row["expected"]["should_anonymize"], bool)
        gold = row["gold"]["should_anonymize"]
        self.assertIn("probabilities", gold)
        self.assertIn("true", gold["probabilities"])
        self.assertIn("false", gold["probabilities"])


if __name__ == "__main__":
    unittest.main()
