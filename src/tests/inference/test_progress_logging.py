"""Tests for throttled progress logging (F3) — pure stdlib, no heavy deps."""
import unittest

from progress_logging import should_log_progress


class TestProgressThrottle(unittest.TestCase):
    def test_logs_about_every_5_percent_over_100(self):
        logged = sum(should_log_progress(i, 100) for i in range(1, 101))
        self.assertEqual(logged, 20)  # at 5,10,...,100 — not 100

    def test_always_logs_final_item(self):
        self.assertTrue(should_log_progress(7, 7))

    def test_small_total_logs_every_item(self):
        # total < 20 -> step == 1 -> logs every item (acceptable no-op throttle)
        self.assertTrue(all(should_log_progress(i, 5) for i in range(1, 6)))

    def test_zero_total_is_safe(self):
        self.assertTrue(should_log_progress(0, 0))

    def test_throttle_reduces_volume_at_scale(self):
        total = 29098  # the real corpus size
        logged = sum(should_log_progress(i, total) for i in range(1, total + 1))
        self.assertLess(logged, total // 100)  # ~21 vs 29k


if __name__ == "__main__":
    unittest.main()
