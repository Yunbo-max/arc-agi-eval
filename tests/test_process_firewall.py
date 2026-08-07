import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from arc_agi_eval.firewall import generate_challenge_tree


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProcessFirewallTests(unittest.TestCase):
    def test_label_mutation_does_not_change_subprocess_prediction_bytes(self) -> None:
        task = {
            "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
            "test": [{"input": [[3, 4]], "output": [[4, 3]]}],
        }
        changed = copy.deepcopy(task)
        changed["test"][0]["output"] = [[9, 9]]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_bytes = []
            challenge_bytes = []
            visible_manifest_bytes = []
            for name, labeled_task in (("original", task), ("changed", changed)):
                labeled = root / name / "labeled"
                labeled.mkdir(parents=True)
                (labeled / "task0001.json").write_text(
                    json.dumps(labeled_task), encoding="utf-8"
                )
                challenge = root / name / "challenge"
                generate_challenge_tree(labeled, challenge)
                challenge_bytes.append((challenge / "task0001.json").read_bytes())
                visible_manifest_bytes.append((challenge / "MANIFEST").read_bytes())

                predictions = root / name / "predictions.json"
                metadata = root / name / "run.json"
                environment = {
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": str(PROJECT_ROOT),
                    "PYTHONHASHSEED": "0",
                }
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "arc_agi_eval",
                        "baseline",
                        str(challenge),
                        "--output",
                        str(predictions),
                        "--metadata",
                        str(metadata),
                    ],
                    cwd=PROJECT_ROOT,
                    env=environment,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                prediction_bytes.append(predictions.read_bytes())

            self.assertEqual(challenge_bytes[0], challenge_bytes[1])
            self.assertEqual(visible_manifest_bytes[0], visible_manifest_bytes[1])
            self.assertEqual(prediction_bytes[0], prediction_bytes[1])


if __name__ == "__main__":
    unittest.main()
