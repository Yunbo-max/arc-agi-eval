import hashlib
import json
from pathlib import Path
import unittest


from arc_agi_eval.validation import load_task


ROOT = Path(__file__).resolve().parents[1]
VIEW_ROOT = (
    ROOT
    / "reports"
    / "e0-challenge-data"
    / "20260806-locked-public-challenge-trees-draft-retry2"
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class LockedPublicChallengeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (VIEW_ROOT / "manifest.json").read_text(encoding="utf-8")
        )

    def test_manifest_digest_and_denominators(self) -> None:
        payload = dict(self.manifest)
        declared = payload.pop("challenge_manifest_sha256")
        self.assertEqual(canonical_sha256(payload), declared)
        self.assertEqual(self.manifest["summary"]["task_count"], 520)
        self.assertEqual(self.manifest["summary"]["test_input_count"], 586)
        self.assertEqual(self.manifest["summary"]["test_output_fields_present"], 0)

    def test_every_declared_file_is_hash_verified_and_label_free(self) -> None:
        observed_tasks = 0
        observed_inputs = 0
        for benchmark_id, view in self.manifest["views"].items():
            records = view["files"]
            challenge_records = [
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"source_path", "source_sha256"}
                }
                for record in records
            ]
            self.assertEqual(
                canonical_sha256(challenge_records), view["challenge_tree_sha256"]
            )
            self.assertEqual(len(records), view["task_count"], benchmark_id)
            self.assertTrue(view["source_hashes_verified"])
            visible_manifest = VIEW_ROOT / view["visible_manifest"]["path"]
            visible_payload = visible_manifest.read_text(encoding="utf-8")
            self.assertNotIn("third_party/", visible_payload)
            self.assertNotIn(str(ROOT), visible_payload)
            self.assertFalse(
                view["visible_manifest"]["contains_labeled_source_locator"]
            )
            for record in records:
                path = VIEW_ROOT / record["path"]
                self.assertTrue(path.is_file(), path)
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"]
                )
                source = ROOT / record["source_path"]
                self.assertEqual(
                    hashlib.sha256(source.read_bytes()).hexdigest(),
                    record["source_sha256"],
                )
                task = load_task(path, require_test_outputs=False)
                self.assertTrue(
                    all("output" not in pair for pair in task["test"]), path
                )
                self.assertEqual(len(task["test"]), record["test_input_count"])
                observed_tasks += 1
                observed_inputs += len(task["test"])
        self.assertEqual(observed_tasks, 520)
        self.assertEqual(observed_inputs, 586)


if __name__ == "__main__":
    unittest.main()
