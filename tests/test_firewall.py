import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from arc_agi_eval.firewall import FirewallError, challenge_only, generate_challenge_tree
from arc_agi_eval.validation import load_task


FIXTURES = Path(__file__).parent / "fixtures"


class FirewallTests(unittest.TestCase):
    def test_challenge_removes_only_test_outputs(self) -> None:
        labeled = load_task(FIXTURES / "data" / "evaluation" / "multi.json")
        challenge = challenge_only(labeled)
        self.assertEqual(challenge["train"], labeled["train"])
        self.assertEqual(
            [pair["input"] for pair in challenge["test"]],
            [pair["input"] for pair in labeled["test"]],
        )
        self.assertTrue(all("output" not in pair for pair in challenge["test"]))

    def test_tree_is_label_free_and_manifest_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "challenge"
            manifest = generate_challenge_tree(FIXTURES / "data", destination)
            self.assertEqual(manifest["tasks_total"], 1)
            record = manifest["files"][0]
            payload = (destination / record["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"])
            generated = json.loads(payload)
            self.assertTrue(all("output" not in pair for pair in generated["test"]))
            visible_manifest = json.loads(
                (destination / "MANIFEST").read_text(encoding="utf-8")
            )
            self.assertEqual(visible_manifest["source_id"], "redacted")
            self.assertNotIn("source", visible_manifest)
            self.assertNotIn(str((FIXTURES / "data").resolve()), json.dumps(visible_manifest))

    def test_rejects_destination_inside_source_or_nonempty(self) -> None:
        source = FIXTURES / "data"
        with self.assertRaisesRegex(FirewallError, "outside"):
            generate_challenge_tree(source, source / "generated")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            (destination / "existing").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(FirewallError, "not empty"):
                generate_challenge_tree(source, destination)


if __name__ == "__main__":
    unittest.main()
