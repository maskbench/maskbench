"""Tests for the inference_times.json schema: normalisation, migration, and the
save/load round-trip across all three on-disk shapes (legacy-flat / mixed / nested).

The mixed shape is the one written by the pre-fix `save_inference_time`; the legacy
shape is what every real checkpoint on disk (incl. the EnvisionGestureChallenge corpus)
currently holds. Both must read and resume without crashing or corrupting.
"""
import os
import json
import tempfile
import unittest

from checkpointer import Checkpointer, estimator_timings, migrate_inference_times

EST = "MediaPipePoseWorldLandmarker"

LEGACY = {EST: {"a": 1.0, "b": 2.0}}                       # flat, no bookkeeping keys
MIXED = {                                                  # bookkeeping + estimator, no "estimators"
    "metadata": {"total_videos": 3, "total_time_taken": 6.0},
    EST: {"a": 1.0, "b": 2.0, "c": 3.0},
    "videos_processed_per_estimator": {EST: 3},
    "total_time_per_estimator": {EST: 6.0},
}


def _clone(d):
    return json.loads(json.dumps(d))


def _checkpointer(checkpoint_dir, total_videos=10):
    """A Checkpointer that uses only checkpoint_dir + total_videos, without the
    /output-creating __init__ (so tests stay isolated to a temp dir)."""
    c = Checkpointer.__new__(Checkpointer)
    c.checkpoint_dir = checkpoint_dir
    c.total_videos = total_videos
    return c


class TestEstimatorTimings(unittest.TestCase):
    def test_legacy_flat_passthrough(self):
        self.assertEqual(estimator_timings(_clone(LEGACY)), LEGACY)

    def test_mixed_strips_reserved_keys(self):
        self.assertEqual(set(estimator_timings(_clone(MIXED)).keys()), {EST})

    def test_nested_returns_submap(self):
        nested = {"estimators": {EST: {"v": 1.0}}, "metadata": {}}
        self.assertEqual(estimator_timings(nested), {EST: {"v": 1.0}})


class TestMigrate(unittest.TestCase):
    def test_legacy_recompute_rollups(self):
        m = migrate_inference_times(_clone(LEGACY), total_videos=10)
        self.assertEqual(m["videos_processed_per_estimator"][EST], 2)
        self.assertAlmostEqual(m["total_time_per_estimator"][EST], 3.0)
        self.assertEqual(set(m["estimators"].keys()), {EST})
        self.assertEqual(m["schema_version"], 1)

    def test_mixed_no_fold_in(self):
        m = migrate_inference_times(_clone(MIXED), total_videos=3)
        self.assertEqual(set(m["estimators"].keys()), {EST})           # not polluted
        self.assertAlmostEqual(m["total_time_per_estimator"][EST], 6.0)  # not 6+6+3

    def test_empty_is_valid(self):
        m = migrate_inference_times({}, total_videos=5)
        self.assertEqual(m["estimators"], {})
        self.assertEqual(m["metadata"]["total_videos"], 5)

    def test_idempotent(self):
        m1 = migrate_inference_times(_clone(MIXED), 3)
        m2 = migrate_inference_times(_clone(m1), 3)
        self.assertEqual(m1["estimators"], m2["estimators"])
        self.assertEqual(m1["total_time_per_estimator"], m2["total_time_per_estimator"])


class TestSaveLoadRoundTrip(unittest.TestCase):
    def _write(self, d, payload):
        with open(os.path.join(d, "inference_times.json"), "w") as f:
            json.dump(payload, f)

    def _read(self, d):
        with open(os.path.join(d, "inference_times.json")) as f:
            return json.load(f)

    def test_resume_legacy_checkpoint_does_not_crash(self):
        """THE regression: saving into a pre-existing LEGACY file used to KeyError."""
        with tempfile.TemporaryDirectory() as d:
            self._write(d, LEGACY)
            _checkpointer(d).save_inference_time(EST, "c", 3.0)
            out = self._read(d)
            self.assertEqual(out["videos_processed_per_estimator"][EST], 3)
            self.assertAlmostEqual(out["metadata"]["total_time_taken"], 6.0)
            self.assertEqual(out["schema_version"], 1)

    def test_load_returns_clean_estimator_map_for_mixed(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, MIXED)
            self.assertEqual(set(_checkpointer(d).load_inference_times().keys()), {EST})

    def test_fresh_save_builds_current_schema(self):
        with tempfile.TemporaryDirectory() as d:
            _checkpointer(d, total_videos=7).save_inference_time("E", "v1", 1.5)
            data = self._read(d)
            self.assertEqual(data["metadata"]["total_videos"], 7)
            self.assertEqual(data["estimators"]["E"]["v1"], 1.5)
            self.assertEqual(data["videos_processed_per_estimator"]["E"], 1)

    def test_overwrite_does_not_double_count(self):
        with tempfile.TemporaryDirectory() as d:
            c = _checkpointer(d)
            c.save_inference_time("E", "v1", 1.0)
            c.save_inference_time("E", "v1", 4.0)   # overwrite
            data = self._read(d)
            self.assertEqual(data["videos_processed_per_estimator"]["E"], 1)
            self.assertAlmostEqual(data["total_time_per_estimator"]["E"], 4.0)

    def test_corrupt_json_loads_empty(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "inference_times.json"), "w") as f:
                f.write("{ not valid json")
            self.assertEqual(_checkpointer(d).load_inference_times(), {})


class TestCheckpointerInit(unittest.TestCase):
    """F2: the 3-arg Checkpointer(dataset_name, total_videos, checkpoint_name) must
    bind correctly — a string must not land in total_videos (the raw_masked_experiment bug)."""

    @unittest.skipUnless(os.path.isdir("/output") and os.access("/output", os.W_OK),
                         "needs a writable /output (container)")
    def test_load_branch_binds_args_not_swapped(self):
        name = "test-f2-init-tmp"
        path = os.path.join("/output", name)
        os.makedirs(path, exist_ok=True)
        try:
            c = Checkpointer("DS", 0, name)  # (dataset_name, total_videos, checkpoint_name)
            self.assertEqual(c.total_videos, 0)
            self.assertEqual(c.dataset_name, "DS")
            self.assertTrue(c.checkpoint_dir.endswith(name))
            self.assertTrue(c.load_checkpoint)
        finally:
            os.rmdir(path)


if __name__ == "__main__":
    unittest.main()
