# Post-run fix plan — MaskBench challenge robustness

Status: **plan only.** Do **not** start any `src/` change until the live run
`EnvisionGestureChallenge-20260608-143736` has finished — the container live-mounts `./src`, so
editing source mid-run can change behaviour under the running process. Web-only work is technically
safe but is also held per instruction.

Audit basis: audit #3. Production signal at time of writing — **1,031 attempted → 1,023 npz → 8
dropped (~0.78%)**, all `Ecolang` Gesture clips; extrapolates to **~225 lost clips** over the 29,098-video run.

---

## 0. Gate / pre-flight (before any code change)

1. **Confirm the run is done** — `main.run()` printed `Done`, or `docker compose` exited; no process holds `./src`.
2. **Snapshot outputs** — copy/zip `…/EnvisionGestureChallenge-20260608-143736/` (especially `poses/`,
   `npz/`, `*_maskbench.log`, `inference_times.json`) before touching anything. The `poses/` JSON are the
   recovery source for the dropped clips (§1).
3. **Branch off `main`, not `uihack`.** These are source fixes, separate from the UI prototype:
   `git checkout main && git checkout -b fix/npz-ragged-and-robustness`. Keep the UI prototype on `uihack`;
   do not entangle the two PRs.
4. **Establish the baseline drop list**: `comm`/set-diff of `poses/<est>/*_poses.json` stems vs `npz/<est>/*.npz`
   stems → the exact clips to recover. Cross-check against the `ERROR` lines in the run log.

---

## 1. A1 — Fix the ragged-npz silent drop  ★ highest priority

**Problem.** `Checkpointer.save_npz` ([checkpointer.py:195-207](../src/checkpointer.py#L195-L207)) builds
`world_landmarks_array` / `image_landmarks_array` as `np.array([... per-frame ...], dtype=float)`. Frames
with no detection yield `[]` while detected frames yield 33 keypoints → **inhomogeneous shape** → `ValueError`,
swallowed by [inference_engine.py:88](../src/inference/inference_engine.py#L88). The pose JSON is written
first, so the clip survives in `poses/` but disappears from `npz/`.

**Fix.** Pad missing frames with `NaN`, exactly as the hand path already does
([checkpointer.py:92-128](../src/checkpointer.py#L92-L128)):
- Factor a helper `body_array(frames, dims)` mirroring `return_hand_world_landmarks`: determine
  `num_keypoints` from the first non-empty frame (fallback 33); for empty/`None` frames emit
  `[[nan]*dims] * num_keypoints`; for present frames map `kp.x/y[/z]` with `None → nan`.
- Apply to both `world_body_landmarks` (dims=3) and `image_body_landmarks` (dims=2).
- Guarantee final shape `(n_frames, K, dims)` with no ragged rows.

**Test (new — this is the regression that would have caught the 225).**
`tests/inference/test_save_npz.py`:
- Build a `VideoPoseResult` of N frames where a subset have `persons=[]` (and some have `None` keypoints).
- Call `save_npz`; assert the file exists, `world_body_landmarks.shape == (N, 33, 3)`,
  `image_body_landmarks.shape == (N, 33, 2)`, empty frames are all-`NaN`, present frames finite.
- A second case: single fully-undetected clip still writes a valid all-`NaN` npz (not skipped).

**Risk.** Low. Changes only array assembly; downstream consumers already expect `NaN` for missing (challenge
spec §5). Verify the `envisionhgdetector` consumer tolerates all-`NaN` frames.

---

## 1b. Recover the already-lost clips — **no re-inference needed**

**Why it's possible.** `VideoPoseResult.to_json` serialises **all** `FramePoseResult` fields via
`asdict` — including `persons_world_landmark`, `hands`, `hands_world_landmark` and `z`
([pose_result_class.py:111-118](../src/pose_result_class.py#L111-L118)). Only `from_json` is lossy
(reads 2D `persons` only). So the full 3D + hand data for every dropped clip **already exists** on disk in
`poses/<est>/<clip>_poses.json`.

**Deliverable.** `scripts/recover_npz_from_poses.py` (one-off, host or container):
1. Diff `poses/` vs `npz/` to get the missing stems.
2. For each: `json.load` the pose file directly (not `from_json`), rebuild the frame objects with full
   world/hand/z fields, and call the **fixed** `save_npz` logic.
3. Write the recovered `npz/<est>/<clip>.npz`; log a recovery manifest.
4. Re-run the `poses ↔ npz` reconciliation → expect 0 drops.

**Verification.** Counts equal; spot-check one recovered npz against its pose JSON for shape + a known frame.

---

## 2. A2 — `parse_filename` never returns `None`

**Problem.** `parse_filename` returns `None` on off-pattern names
([utils.py:175-190](../src/utils.py#L177)); callers unpack `.values()` unguarded
([checkpointer.py:179](../src/checkpointer.py#L179), [pose_renderer.py:187](../src/rendering/pose_renderer.py#L187))
→ `AttributeError`. In rendering (which your run hasn't reached yet) this kills the whole video's render.

**Fix.**
- `parse_filename` always returns a dict with the six keys; on failure set `corpus=clip_id=…` to a safe
  fallback (`speaker="NA"`, `category="NA"`, etc.) and add `parsed: bool`.
- Callers: if `not parsed`, skip metadata stamping in npz / use the raw stem as `clip_id` in the render
  caption — never crash.
- Align with the metadata standard: this is the *seed*; the manifest (§5) is the source of truth.

**Test.** `tests/test_parse_filename.py` — `clip_final_v2`, `a_b`, empty, and the real tricky names
(`SAGAplus_V8K2_35_Gesture_iconic_deictic`, `ZHUBO_9_070_26_Gesture_NA`, `MULTISIMO_S07_0007_gesture_move`)
return safe dicts with expected fields; `save_npz`/render don't raise.

**Risk.** Low. Strictly widens accepted input.

---

## 3. A3 — `fps = 0` guard  *(decision: skip + record)*

**Problem.** `int(cap.get(CAP_PROP_FPS))` can be 0; `format_frame` then divides by it
([mediapipe_worldlandmarker_pose_estimator.py:200](../src/models/mediapipe_worldlandmarker_pose_estimator.py#L200)),
crashing and silently dropping the clip.

**Fix (decided).** In `get_video_metadata`, if `fps <= 0`, raise a typed `UnprocessableVideoError` caught by the
engine and recorded as a first-class **`unprocessable`** outcome in the run manifest (§4) — visible, with a
stated reason, never a silent drop.

**Test.** Metadata read with `fps=0` produces the typed error / recorded `unprocessable` outcome, not a bare
`ZeroDivisionError`.

---

## 4. Structured logging contract (`log.jsonl` + `run.json`)

Makes `runs.html` / `build_run_report.ps1` real and ends the silent-failure problem.

**`<run_dir>/log.jsonl`** — one JSON object per line:
`{ts, level, stage, estimator, video, message}` where `stage ∈ {config, dataset, inference, npz, render, eval}`.
- Add a small `RunLogger` (writes JSONL + mirrors to the existing text log).
- Route the `print()` / `logging.*` calls in [inference_engine.py](../src/inference/inference_engine.py) and
  [checkpointer.py](../src/checkpointer.py) through it.
- On caught exceptions, log `repr(e)` **and** `traceback.format_exc()` (not bare `str(e)` as today).

**`<run_dir>/run.json`** — manifest read by the UI (no log-parsing):
`{run_id, status, started, ended, config_sha256, estimators, counts:{discovered, attempted, npz, dropped,
unparsed, unprocessable}, per_video:[{clip_id, status, reason, seconds}]}`.

**Wire-up.** Point `build_run_report.ps1` at `run.json`/`log.jsonl` when present (fall back to the text-log
parser otherwise). Replace `runs.html` canned `LOG` array read with these once available.

**Risk.** Medium — touches the engine/checkpointer hot path. Keep logging cheap (append, no locks beyond the
existing `inference_times` lock). Coordinate with A1/A2/A3 since they edit the same files — sequence A1→A2→A3
then logging, or do logging last on top.

---

## 5. Metadata manifest = source of truth (follow-up, depends on A2 + METADATA_STANDARD.md)

**Scope.** MaskBench writes `<run_dir>/metadata.csv` seeded from `parse_filename`, with controlled-vocab
**normalisation** per [METADATA_STANDARD.md §3](../web/METADATA_STANDARD.md): canonical corpus casing
(`SAGA→SaGA`), subtype casing (lowercase adjectives, keep Ecolang names), single `NA` null sentinel, typo fix
`deictic__abstract→deictic_abstract`; plus `speaker_id` corpus-prefixed for global uniqueness, `is_mirror` +
`original_clip_id` linkage, and `source_sha256`. Validation enforces `subtype ∈ vocab[corpus]` (§3.1). The
Builder UI's Step 5 already mocks the review/override + gate; this is the backend that produces what it reviews.

**Subtype vocabularies are now pinned** in METADATA_STANDARD §3.1 (extracted from the data, normalisation
decided). No longer blocking.

**Defer** behind A1–A4; it's the largest change.

---

## 6. Web-only (safe anytime; held per instruction)

- `build_run_report.ps1`: default `$avgS/$maxS/$minS` to 0 when `inference_times.json` is read mid-write so a
  race can't null-crash the header `[math]::Round(...)`. Make the corpus key fall back to `"(unparsed)"` for
  off-pattern stems. ~10 min, no run impact.

---

## 7. Sequencing, verification, PRs

1. **Branch** `fix/npz-ragged-and-robustness` off `main`.
2. **Order:** A1 (+ test) → 1b recovery script (run it, reconcile to 0 drops) → A2 (+ test) → A3 (+ test) →
   logging contract. Metadata manifest (§5) as a **separate** PR.
3. **Verify each** on a *small* re-run (e.g. a 20-clip subset incl. a known ragged clip + an off-pattern name +
   an `fps=0` clip) — do **not** re-run all 29k. Recovery (§1b) needs no inference at all.
4. **Tests green** (`pytest`); the new regression tests are the acceptance bar for A1/A2.
5. **Two PRs:** (a) robustness fixes + recovery + tests; (b) logging contract + UI wire-up. Keep `uihack`
   (prototype) independent.
6. **Rollback:** each fix is self-contained; the recovery script is additive (writes only missing npz). Keep the
   pre-fix output snapshot until the recovered dataset is validated end-to-end by the `envisionhgdetector` consumer.

---

## Decisions (resolved 2026-06-08)

- **A3 — `fps=0`:** skip + record as `unprocessable` in the manifest (not silent). ✓
- **Subtype vocabularies:** extracted from the data and pinned in METADATA_STANDARD §3.1. Casing = lowercase
  adjectives but keep Ecolang's CamelCase taxonomy; null sentinel = single `NA`. ✓
- **`Move` class:** moot — **no `Move` label exists in the data** (dataset is 2-class, balanced 14,549/14,549;
  `MoveClips/` empty). `move` is a SaGA *subtype* only. Reserve `Move` only if such data is added later. ✓

## Data corrections surfaced by the 2026-06-08 filename scan (fold into the work)

- **2-class, not 3-class** — update the Builder mock's class-distribution panel (currently shows a fictitious
  `Move 2%`) and any 3-class assumptions in scoring to 2-class balanced. *(UI-only; held with everything else.)*
- **Corpus casing** corrected to actual prefixes: `Ecolang, TedM3D, SAGAplus, Multisimo, ZHUBO, GESRES, SaGA`.
- **Three null sentinels** (`None`/`NA`/`N_A`) in the source filenames → normalise to one `NA`.
- **Source typo** `deictic__abstract` (GESRES, double underscore) → `deictic_abstract` in normalisation.
- **`speaker_id` uniqueness** — prefixes like `ad01`, `S07`, `V8K2`, `9_070` are not globally unique; the manifest
  must corpus-prefix them (`Ecolang_ad01`) so the speaker-independent split can't leak across corpora.
