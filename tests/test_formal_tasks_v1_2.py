import json
import tempfile
import unittest
from pathlib import Path

from group_llm_benchmark.tasks.bbh import normalize_target
from group_llm_benchmark.tasks.formal_mcq import FormalMultipleChoiceTask
from group_llm_benchmark.tasks.numeric_qa import NumericQATask, canonical_number


class FormalTaskTests(unittest.TestCase):
    def test_numeric_canonicalization(self):
        self.assertEqual(canonical_number("1,234.00"), "1234")
        self.assertEqual(canonical_number("-0.50"), "-0.5")
        self.assertIsNone(canonical_number("not a number"))

    def test_bbh_normalization_preserves_brackets(self):
        self.assertEqual(normalize_target("(A)"), "a")
        self.assertEqual(normalize_target("] } )"), "] } )")
        self.assertNotEqual(normalize_target("] } )"), normalize_target(") } ]"))

    def test_formal_mcq_supports_labels_beyond_f(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.jsonl"
            path.write_text(json.dumps({
                "id": "q1", "question": "Pick H", "choices": {letter: letter for letter in "ABCDEFGH"},
                "answer": "H", "source": "test"
            }) + "\n", encoding="utf-8")
            task = FormalMultipleChoiceTask({"dataset_path": str(path), "name": "test"})
            sample = task.load_samples()[0]
            self.assertEqual(task.parse_response("Answer: H", sample), "H")
            self.assertEqual(task.score("H", sample)["accuracy"], 1.0)

    def test_numeric_task_extracts_last_number(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.jsonl"
            path.write_text(json.dumps({
                "id": "q1", "input": "1+1?", "reference": "2", "source": "test"
            }) + "\n", encoding="utf-8")
            task = NumericQATask({"dataset_path": str(path), "name": "test"})
            sample = task.load_samples()[0]
            self.assertEqual(task.parse_response("Work: 1 + 1. Final answer: 2", sample), "2")
            self.assertEqual(task.score("2", sample)["numeric_exact_match"], 1.0)


if __name__ == "__main__":
    unittest.main()

