"""Tests for shared benchmark CLI helpers."""

import logging
import unittest

from compass.benchmark.cli import (
    log_and_exit,
    log_errors_and_exit,
    require_or_exit,
    run_or_exit,
)


class BenchmarkCliTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("compass.benchmark.cli.tests")

    def test_log_and_exit_logs_message_and_exits(self):
        with self.assertLogs(self.logger, level="ERROR") as logs:
            with self.assertRaises(SystemExit) as exc:
                log_and_exit(self.logger, "bad config", exit_code=2)

        self.assertEqual(exc.exception.code, 2)
        self.assertEqual(logs.output, ["ERROR:compass.benchmark.cli.tests:bad config"])

    def test_require_or_exit_passes_when_condition_is_true(self):
        require_or_exit(True, self.logger, "should not fail", exit_code=2)

    def test_require_or_exit_logs_and_exits_when_condition_is_false(self):
        with self.assertLogs(self.logger, level="ERROR") as logs:
            with self.assertRaises(SystemExit) as exc:
                require_or_exit(False, self.logger, "missing models", exit_code=1)

        self.assertEqual(exc.exception.code, 1)
        self.assertEqual(logs.output, ["ERROR:compass.benchmark.cli.tests:missing models"])

    def test_log_errors_and_exit_logs_all_messages_and_exits(self):
        with self.assertLogs(self.logger, level="ERROR") as logs:
            with self.assertRaises(SystemExit) as exc:
                log_errors_and_exit(
                    self.logger,
                    ["bad row 1", "bad row 2"],
                    exit_code=1,
                )

        self.assertEqual(exc.exception.code, 1)
        self.assertEqual(
            logs.output,
            [
                "ERROR:compass.benchmark.cli.tests:bad row 1",
                "ERROR:compass.benchmark.cli.tests:bad row 2",
            ],
        )

    def test_run_or_exit_returns_callback_value(self):
        value = run_or_exit(lambda: "ok", self.logger, exit_code=1)
        self.assertEqual(value, "ok")

    def test_run_or_exit_logs_value_error_and_exits(self):
        with self.assertLogs(self.logger, level="ERROR") as logs:
            with self.assertRaises(SystemExit) as exc:
                run_or_exit(
                    lambda: (_ for _ in ()).throw(ValueError("bad row")),
                    self.logger,
                    exit_code=1,
                )

        self.assertEqual(exc.exception.code, 1)
        self.assertEqual(logs.output, ["ERROR:compass.benchmark.cli.tests:bad row"])


if __name__ == "__main__":
    unittest.main()
