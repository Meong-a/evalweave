import json
import tempfile
import unittest
from pathlib import Path

from group_llm_benchmark.config import load_experiment
from group_llm_benchmark.tasks.factory import create_task


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DomainExtensionV122Tests(unittest.TestCase):
    def load_template(self, filename: str):
        path = PROJECT_ROOT / "configs/task_templates_v1_2_2" / filename
        config = json.loads(path.read_text(encoding="utf-8"))
        task = create_task(config, base_dir=path.parent)
        return task, task.load_samples()

    def test_medical_mcq_template(self):
        task, samples = self.load_template("medical_mcq_task_v1_2_2.json")
        self.assertEqual(len(samples), 2)
        self.assertEqual(task.score("B", samples[0])["accuracy"], 1.0)

    def test_summary_template(self):
        task, samples = self.load_template("summary_task_v1_2_2.json")
        self.assertEqual(len(samples), 2)
        metrics = task.score(samples[0].reference["text"], samples[0])
        self.assertAlmostEqual(metrics["rouge_l"], 1.0)

    def test_custom_plugin_without_factory_edit(self):
        task, samples = self.load_template("custom_keyword_task_v1_2_2.json")
        self.assertEqual(len(samples), 2)
        metrics = task.score("建议改善饮食并坚持运动。", samples[0])
        self.assertEqual(metrics["keyword_recall"], 1.0)
        self.assertEqual(metrics["all_keywords"], 1.0)

    def test_task_reference_resolves_dataset_relative_to_task_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            (root / "tasks").mkdir()
            (root / "models").mkdir()
            dataset = root / "data/items.jsonl"
            dataset.write_text(json.dumps({
                "id": "q1", "question": "Pick A", "choices": {"A": "yes", "B": "no"},
                "answer": "A", "source": "test"
            }) + "\n", encoding="utf-8")
            model = root / "models/model.json"
            model.write_text(json.dumps({"type": "mock", "id": "model"}), encoding="utf-8")
            task = root / "tasks/task.json"
            task.write_text(json.dumps({
                "type": "mcq_formal", "name": "task", "dataset_path": "../data/items.jsonl"
            }), encoding="utf-8")
            experiment = root / "experiment.json"
            experiment.write_text(json.dumps({
                "schema_version": "1.0", "experiment_name": "relative-path-test",
                "model_ref": "models/model.json", "task_ref": "tasks/task.json"
            }), encoding="utf-8")
            loaded = load_experiment(experiment)
            self.assertEqual(Path(loaded.task["dataset_path"]), dataset.resolve())


if __name__ == "__main__":
    unittest.main()
