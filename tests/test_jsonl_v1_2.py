import json
import tempfile
import unittest
from pathlib import Path

from group_llm_benchmark.utils.jsonl_v1_2 import read_jsonl


class JsonlV12Tests(unittest.TestCase):
    def test_unicode_line_separator_is_preserved_inside_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            records = [
                {"id": "one", "text": "left\u2028right"},
                {"id": "two", "text": "normal"},
            ]
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.assertEqual(read_jsonl(path), records)


if __name__ == "__main__":
    unittest.main()
