# MaskBench UI (prototype) — `uihack`

A UI to make MaskBench usable for SSH researchers building **challenge datasets**, in the
**MaskingOPS** design language (Carbon-faithful: IBM Plex + Carbon color tokens, no `@carbon/*`
dependency yet). Everything here is **static and read-only** — it imports no Python, touches no
Docker, and never writes to a dataset or output dir. Safe to open next to a live run.

## Contents

| File | What it is | Status |
|---|---|---|
| `challenge-builder.html` | 7-step guided wizard: source → extraction → run → **quality/integrity** → labels+provenance → split → export. Replaces hand-edited YAML/`.env`. | clickable mockup |
| `runs.html` | MaskBench **Run Log** home: runs list, reconciliation, filterable structured log stream, per-video table. | clickable mockup (canned data) |
| `build_run_report.ps1` | **Read-only** generator that renders `run-report.html` from a *real* run dir. | working |
| `run-report.html` | Generated output (git-ignored — contains run-specific data). | generated |
| `METADATA_STANDARD.md` | Proposed metadata rules & standards for releasable challenge datasets. | draft |

## View the mockups

Open the `.html` files in any browser (no build step):

```powershell
start web\challenge-builder.html
start web\runs.html
```

The MaskBench nav link in the builder cross-links to the run log.

## Generate a real run report

`runs.html` is a mock. To see **real** data from an actual run, use the generator (host-native
PowerShell, pure reads — safe while a run is in progress):

```powershell
pwsh web\build_run_report.ps1                 # latest run; paths from .env
pwsh web\build_run_report.ps1 -Run <folder>   # a specific checkpoint
pwsh web\build_run_report.ps1 -SkipDatasetCount   # skip the dataset scan (faster)
start web\run-report.html
```

It reads the run's `*_maskbench.log`, `inference_times.json`, `poses/`, `npz/`, `config.yml`, and
(optionally) the dataset dir, then writes `run-report.html`. Re-run to refresh; ~0.7s incl. a 29k-file
dataset scan.

## The logging contract (the real fix behind the mock)

The mockups assume MaskBench emits structured, machine-readable run telemetry. The minimal,
low-risk `src/` change to make `runs.html` real:

1. **`<run_dir>/log.jsonl`** — one event per line: `{ts, level, stage, estimator, video, message}`.
   Route the existing `print()` / `logging` calls in
   [`inference_engine.py`](../src/inference/inference_engine.py) and
   [`checkpointer.py`](../src/checkpointer.py) through it. Include `repr(e)` + traceback on caught
   exceptions, not bare `str(e)`.
2. **`<run_dir>/run.json`** — manifest: status, started/ended, config hash, per-video outcome
   (so the runs list + reconciliation are reads, not log-parsing).
3. The UI tails `log.jsonl` and reads `run.json`.

Until then, `build_run_report.ps1` parses the existing plain-text log as a stopgap — which already
surfaces real failures (e.g. the ragged-npz drops where `poses/` count > `npz/` count).

## Notes

- Prototype only; no backend. `src/` is **not** modified here (a live run mounts `./src`).
- Metadata rigor: see [`METADATA_STANDARD.md`](METADATA_STANDARD.md).
