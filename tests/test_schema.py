import unittest
from pathlib import Path

from group_llm_benchmark.schemas import ExperimentConfig, Sample


class SchemaTests(unittest.TestCase):
    def test_sample_requires_capability(self):
        with self.assertRaises(ValueError):
            Sample("x", "task", (), {}, {})

    def test_experiment_config(self):
        config = ExperimentConfig.from_dict(
            {
                "schema_version": "1.0",
                "experiment_name": "demo",
                "model": {"type": "mock"},
                "task": {"type": "mcq"},
                "output_root": str(Path("runs").resolve()),
            }
        )
        self.assertEqual(config.batch_size, 1)

    def test_rejects_unknown_schema(self):
        with self.assertRaises(ValueError):
            ExperimentConfig.from_dict(
                {
                    "schema_version": "9.9",
                    "experiment_name": "demo",
                    "model": {},
                    "task": {},
                }
            )


if __name__ == "__main__":
    unittest.main()
