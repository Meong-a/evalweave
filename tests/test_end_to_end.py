import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from group_llm_benchmark.analysis import build_leaderboard, build_suite_matrix, compare_runs
from group_llm_benchmark.config import load_experiment
from group_llm_benchmark.runners import BenchmarkRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EndToEndTests(unittest.TestCase):
    def test_mock_runs_and_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / "runs"
            baseline_config = replace(
                load_experiment(PROJECT_ROOT / "configs/experiments/mock_baseline_public_demo_v1.json"),
                output_root=output_root,
            )
            candidate_config = replace(
                load_experiment(PROJECT_ROOT / "configs/experiments/mock_candidate_public_demo_v1.json"),
                output_root=output_root,
            )
            baseline_run = BenchmarkRunner(baseline_config).run()
            candidate_run = BenchmarkRunner(candidate_config).run()
            baseline_metrics = json.loads((baseline_run / "metrics_v1.json").read_text(encoding="utf-8"))
            candidate_metrics = json.loads((candidate_run / "metrics_v1.json").read_text(encoding="utf-8"))
            self.assertEqual(baseline_metrics["count"], 4)
            self.assertAlmostEqual(baseline_metrics["primary_score"], 0.5)
            self.assertAlmostEqual(candidate_metrics["primary_score"], 1.0)
            comparison = compare_runs(baseline_run, candidate_run, Path(temp) / "comparisons", bootstrap_samples=100)
            result = json.loads((comparison / "comparison_v1.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(result["overall"]["delta"], 0.5)
            self.assertEqual(result["transitions"]["improved"], 2)
            leaderboard = build_leaderboard(
                [baseline_run, candidate_run], Path(temp) / "leaderboards", "test_leaderboard_v1"
            )
            board = json.loads((leaderboard / "leaderboard_v1.json").read_text(encoding="utf-8"))
            self.assertEqual(board["ranking"][0]["model_id"], "mock-candidate-v1")
            matrix = build_suite_matrix(
                [baseline_run, candidate_run], Path(temp) / "suite_reports", "test_matrix_v1"
            )
            matrix_data = json.loads((matrix / "suite_matrix_v1.json").read_text(encoding="utf-8"))
            self.assertEqual(len(matrix_data["runs"]), 2)


if __name__ == "__main__":
    unittest.main()
