# Fix plan — `inference_times.json` schema + progress logging

Status: **plan for review (rev. 2 — adversarial audit folded in).** Implements fixes on a branch as a
reviewable PR; Sharjeel reviews the diff against this doc. Scope = the two commits on
`origin/envision_gesture_challenge` (`6f14949` hide ffmpeg output, `06b8381` better progress logging).

First concrete instance of the versioned-schema work — see issues **#153** (versioned schemas) and **#154**
(run manifest).

> **Rev. 2 changelog (what the audit changed):** the earlier plan assumed only two on-disk shapes
> (legacy-flat, new-nested). There is a **third, already-real shape** — Sharjeel's *mixed* file
> (bookkeeping keys **and** estimator keys at top level, **no** `"estimators"` key). The old
> `load = data.get("estimators", data)` one-liner **silently corrupts** that shape. F1 below now normalises
> all three shapes on read **and** in migration. Also fixed: line anchors point at Sharjeel's file; `main.py`
> is explicitly reconciled (it is 2-arg on this branch); backend instructions de-contradicted; eval-plot
> severity re-scoped; secondary bugs flagged.

---

## 0. Pre-flight / merge gate (do before any F-fix)

These fixes target **Sharjeel's code**, which is **not yet on this branch**. Order matters:

1. **Branch** `fix/inference-times-schema-and-logging` off an integration point that contains both
   `6f14949` + `06b8381`, so the PR diff is *only* these fixes.
2. **Resolve the `main.py` conflict deliberately.** This branch's `main.py` has the `run_plan` guard **and**
   is still **2-arg**: `Checkpointer(dataset.name, checkpoint_name)` ([src/main.py:30](../src/main.py#L30)).
   Sharjeel's is **3-arg**: `Checkpointer(dataset.name, len(dataset), checkpoint_name)`
   (`origin/envision_gesture_challenge:src/main.py:28`). **Post-merge `main.py` must keep BOTH** the
   `run_plan` block and the 3-arg call. Verify by grep before continuing — do not assume the merge got it
   right. A 2-arg survivor puts `checkpoint_name` into `total_videos` → `self.total_videos` becomes a string.
3. **Confirm `total_videos` flows in** wherever `Checkpointer` is constructed (see F2).

---

## The contract at the centre of this

`06b8381` changed `inference_times.json` from flat `estimator → {video → seconds}` to a mix of bookkeeping
keys **and** estimator keys at the same top level. That top level is iterated as "estimators" by 4 consumers.

### Three on-disk shapes exist (this is the crux the first plan missed)

| Shape | Looks like | Where it comes from |
|---|---|---|
| **legacy-flat** | `{ "Est": { "v": 1.2 } }` | every real file on disk today (incl. the 0608 corpus) |
| **mixed** | `{ "metadata": {…}, "Est": {…}, "videos_processed_per_estimator": {…}, … }` | Sharjeel's **unfixed** `save_inference_time` (no `"estimators"` key) |
| **nested (target)** | `{ "metadata": {…}, "estimators": { "Est": {…} }, … }` | what this plan writes |

A single discriminator of `"estimators" in data` is **not** enough — it misclassifies *mixed* as *legacy* and
folds the bookkeeping keys into the estimator map. The fix keys off a **RESERVED** set instead.

### Target schema (decided: nest under `estimators`, add `schema_version`)

```jsonc
{
  "schema_version": 1,
  "metadata": { "total_videos": 29098, "total_time_taken": 1234.5 },
  "estimators": { "MediaPipePoseWorldLandmarker": { "<clip>": 1.23 } },
  "videos_processed_per_estimator": { "MediaPipePoseWorldLandmarker": 1023 },
  "total_time_per_estimator":       { "MediaPipePoseWorldLandmarker": 1234.5 }
}
```

```python
RESERVED = {"schema_version", "metadata", "estimators",
            "videos_processed_per_estimator", "total_time_per_estimator"}
```

---

## F1 — Normalise all three shapes on read + migrate on write  ★ root cause

**Problem.** (a) Bookkeeping keys leak into estimator iteration. (b) `save_inference_time`
([checkpointer.py:118](../src/checkpointer.py#L118), Sharjeel's version) only creates the bookkeeping keys in
the *new-file* branch, so loading any pre-existing file then doing
`inference_times["videos_processed_per_estimator"]` raises `KeyError`. Verified: the 0608 corpus file is
legacy-flat (top-level keys = `["MediaPipePoseWorldLandmarker"]`).

**Fix — one shared normaliser** (new helper, used by both read and migrate):

```python
def _estimator_timings(data: dict) -> dict:
    """estimator -> {video: seconds}, from any of the 3 on-disk shapes."""
    if "estimators" in data:                              # nested (target)
        return data["estimators"]
    if any(k in data for k in RESERVED):                  # mixed (Sharjeel unfixed)
        return {k: v for k, v in data.items() if k not in RESERVED}
    return data                                           # legacy-flat
```

**Fix — `load_inference_times`** ([checkpointer.py:210](../src/checkpointer.py#L210), Sharjeel's version):
wrap `json.load` in `try/except json.JSONDecodeError` (the read path holds no lock; a mid-write file must not
crash the loader — `build_run_report.ps1:60` already guards this, the Python side does not) and
`return _estimator_timings(data)`. The visualizer/`inference_time_plot` then receive a clean estimator map for
**all three** shapes — **no changes needed in the evaluation code**.

**Fix — `save_inference_time` migration** (write path):

```python
def _migrate(self, data: dict) -> dict:
    if "estimators" in data and "metadata" in data:      # already current
        data.setdefault("videos_processed_per_estimator", {})
        data.setdefault("total_time_per_estimator", {})
        data.setdefault("schema_version", 1)
        return data
    est = _estimator_timings(data)                        # excludes RESERVED → no fold-in
    return {
        "schema_version": 1,
        "metadata": {"total_videos": self.total_videos,
                     "total_time_taken": sum(sum(v.values()) for v in est.values())},
        "estimators": est,
        "videos_processed_per_estimator": {e: len(v) for e, v in est.items()},
        "total_time_per_estimator": {e: sum(v.values()) for e, v in est.items()},
    }
```

Because migration goes through `_estimator_timings`, a **mixed** file does **not** fold `metadata` /
roll-up dicts into `estimators` (the bug the audit caught). Migration is idempotent and recomputes totals
from data (self-healing).

**Consumers updated by F1:**
- `checkpointer.py` — `_estimator_timings`, `load_inference_times`, `save_inference_time` (+ `_migrate`).
- `web/build_run_report.ps1` — read `$it.estimators` if present, else mixed/legacy fallback (mirror the 3-shape
  logic; ~5 lines). Keep the existing mid-write `try/catch`.
- `backend/main.py` — `_inference_times()` returns `_estimator_timings(data)` (the normalised map).
  `dashboard_stats` **keeps its current robust loop** (`for est, vids in _inference_times(r).items(): if
  isinstance(vids, dict): …`) which already works across all runs on disk (all legacy today). **Do not** switch
  it to read `metadata.total_time_taken` directly — that KeyErrors on every existing legacy run. *Optional:* use
  `metadata.total_time_taken` **only when present**, else fall back to the loop. (This removes the rev-1
  contradiction.)

**Eval-plot severity — re-scoped (audit finding).** In the *full* eval pipeline the bogus keys are actually
filtered out before plotting: `sort_inference_times_pose_estimator_order`
([maskbench_visualizer.py:101-103](../src/evaluation/visualizer/maskbench_visualizer.py#L101)) keeps only
estimators present in `metric_results`, so the "3 bogus bars" may **not** reproduce via the visualizer. The
**definite** corruption is in consumers that average the raw dict with **no** sort filter — `build_run_report.ps1`
and `backend/main.py` — plus the **definite** `KeyError` crash-on-resume. F1 still fixes all of these at the
root; we just don't oversell the plot symptom.

**Test.** `tests/inference/test_inference_times.py`:
- Fresh dir → schema has `schema_version/metadata/estimators/…`; `load_inference_times()` returns
  `{est: {video: sec}}`; roll-ups correct.
- **Legacy** file `{"Est": {"v1": 1.0}}` → `save_inference_time("Est","v2",2.0)`: no KeyError; migrated;
  `total_time_per_estimator["Est"] == 3.0`; `videos_processed == 2`.
- **Mixed** file (`metadata` + `Est` + roll-ups, no `"estimators"`) → `load_inference_times()` returns **only**
  `{"Est": …}` (RESERVED stripped); a subsequent save migrates without folding bookkeeping keys into estimators.
- Overwrite same `(est, video)` → counts/totals unchanged.
- Corrupt/truncated JSON → `load_inference_times()` returns `{}` (no exception).

**Risk.** Low–medium. Localised to read/write helpers; reader contracts preserved; migration idempotent.

---

## F2 — `Checkpointer(total_videos)` callers

**Problem.** `06b8381` inserted required positional `total_videos` mid-signature. It updated `main.py` (on its
branch) but not [src/scripts/raw_masked_experiment.py:69](../src/scripts/raw_masked_experiment.py#L69):
`Checkpointer(dataset_name, f"{dataset_name}-{strategy}")` → strategy string lands in `total_videos`,
`checkpoint_name` → `None` → a *new* empty checkpoint instead of loading.

**Fix (primary, keeps Sharjeel's signature).** `Checkpointer(dataset_name, len(dataset), f"{dataset_name}-{strategy}")`.

**Fix (alternative for review).** `def __init__(self, dataset_name, checkpoint_name=None, *, total_videos=None)`
— append as keyword-with-default so no positional caller can silently break; `total_videos` degrades to
`metadata.total_videos: null` when unset. *Recommend primary; flag this for Sharjeel.*

**Heads-up (separate pre-existing bug, not ours to fix here but note it):**
[raw_masked_experiment.py:70](../src/scripts/raw_masked_experiment.py#L70) calls `load_pose_results()` with **no
arg**, but the signature requires `pose_estimator_names`
([checkpointer.py:146](../src/checkpointer.py#L146)). So this script is **already broken** independent of the
signature change — don't let a green F2 unit test imply the script runs end-to-end. Flag it; fix out of scope.

**Test.** Construct `Checkpointer` both in-repo ways; assert `total_videos` is the int and `checkpoint_name` is
the intended value (not swapped).

**Risk.** Low. One-line caller change.

---

## F3 — Progress logging volume

**Problem.** `logging.info(progress.__str__())` runs **every** video
([inference_engine.py](../src/inference/inference_engine.py)) and **every** npz
([npz_format.py](../src/save_formats/npz_format.py)) → ~29k near-duplicate bar-strings to `*_maskbench.log` +
per-iteration I/O. tqdm already renders the live bar on stderr.

**Fix.** Throttle the **file** log to milestones via a `_log_progress(done, total)` helper:
`step = max(1, total // 20)` (≈ every 5%) + a final `Done: N processed, M skipped`. Keep the live tqdm bar.

**Note (audit).** For small runs (`total < 20`), `total // 20 == 0 → step = 1`, i.e. logs every item — a no-op
throttle, which is fine. The real check is the **100-item unit test** (assert ≤ ~21 progress records), not the
≤20-clip verify run.

**Risk.** Low. Logging-only.

---

## F4 — tqdm thread-safety (latent; decision needed)

**Problem.** Per-estimator bars + bare `print()` inside the `ThreadPoolExecutor` (workers = #estimators), no
`position=`. Garbles with >1 estimator. Not hit today (single estimator).

**Fix / caveat (audit).** A real `position=` fix needs an **estimator index threaded through**
`executor.submit` ([inference_engine.py:46-50](../src/inference/inference_engine.py#L46)) — the estimator
object only has `.name`, not an index. `npz_format.create` is sequential, so it needs no `position=`.
**Decision for Sharjeel:** thread the index through and set `position=`/`leave=True`, **or** defer with
`# TODO(threadsafe)` since it's latent. Plan defers unless he wants it now.

**Risk.** Low. Console-only.

---

## F5 — `6f14949` ffmpeg quieting — adopt as-is ✅

`-hide_banner -loglevel error` is correct and self-contained. Merged unmodified.

---

## Sequencing, verification, PR

1. **Merge gate (§0)** — branch with both commits; reconcile `main.py` to keep `run_plan` **and** 3-arg.
2. **Order.** F1 (normaliser + migration + reader updates + tests) → F2 (caller) → F3 → F4 (or defer). F5 via merge.
3. **Verify** on a tiny re-run (≤20 clips) that **resumes a legacy-schema checkpoint** (the KeyError case today):
   confirm no crash, migrated `inference_times.json` (has `schema_version`/`estimators`), eval inference-time
   plot shows one real estimator, and `build_run_report.ps1` + `/api/dashboard/stats` show a sane avg. Also feed
   a hand-written **mixed** file through `load_inference_times` and assert RESERVED keys are stripped. Do **not**
   re-run the full 29k.
4. **Tests green** (`pytest`); the legacy + mixed migration tests (F1) and the caller test (F2) are the bar.
5. **One PR**, this doc linked; commits map 1:1 to F1–F4 so Sharjeel reviews each fix against its section.
6. **Rollback.** Each fix self-contained. Migration is forward-only but non-destructive (recomputes from data);
   keep a copy of one real `inference_times.json` until the eval plot + report are confirmed correct.

---

## Open decisions for Sharjeel

- **F2 signature:** fix the caller (keep his positional `total_videos`) vs. keyword-with-default (defensive,
  changes signature). Plan recommends the former.
- **F4 tqdm:** thread the index through for `position=` now, vs. defer (latent until multi-estimator configs).
- **`schema_version`:** introduce it now on `inference_times.json` (seeds #153) — confirm the value/semantics.
- **Naming:** `estimators` as the nesting key alongside `total_time_per_estimator` / `videos_processed_per_estimator`.
