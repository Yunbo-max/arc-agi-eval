import os
import sys
import tempfile
import unittest
from pathlib import Path

from arc_agi_eval.execution import run_process


class ExecutionTests(unittest.TestCase):
    def test_captures_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            success = run_process(
                [sys.executable, "-c", "print('ready')"],
                cwd=temporary,
                timeout_seconds=2,
            )
            self.assertEqual(success.status, "passed")
            self.assertEqual(success.return_code, 0)
            self.assertEqual(success.stdout, "ready\n")

            failure = run_process(
                [sys.executable, "-c", "import sys; print('bad', file=sys.stderr); sys.exit(7)"],
                cwd=temporary,
                timeout_seconds=2,
            )
            self.assertEqual(failure.status, "failed")
            self.assertEqual(failure.return_code, 7)
            self.assertEqual(failure.stderr, "bad\n")

    def test_timeout_kills_the_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "child-survived"
            code = (
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable,'-c',"
                f"\"import time,pathlib; time.sleep(0.4); pathlib.Path({str(marker)!r}).write_text('x')\"]); "
                "time.sleep(10)"
            )
            result = run_process(
                [sys.executable, "-c", code],
                cwd=temporary,
                timeout_seconds=0.1,
            )
            self.assertEqual(result.status, "timeout")
            self.assertTrue(result.timed_out)
            self.assertIsNone(result.return_code)
            # The process-group kill should prevent the child from reaching its write.
            import time

            time.sleep(0.5)
            self.assertFalse(marker.exists())

    def test_rejects_invalid_arguments(self) -> None:
        with self.assertRaisesRegex(ValueError, "command"):
            run_process([], cwd=os.curdir, timeout_seconds=1)
        with self.assertRaisesRegex(ValueError, "positive"):
            run_process([sys.executable], cwd=os.curdir, timeout_seconds=0)


if __name__ == "__main__":
    unittest.main()
