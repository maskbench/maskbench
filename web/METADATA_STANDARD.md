# MaskBench Challenge Dataset — Metadata Standard (draft v0.1)

Status: **draft / proposal** on the `uihack` branch. This defines the metadata rules a
MaskBench-built *challenge dataset* should follow before public release. It is the spec the
Challenge Builder UI validates against.

---

## 0. Why this exists

Today, all challenge metadata (corpus, speaker, clip, label, subtype, mirror) is **reverse-engineered
from filenames** by a heuristic parser ([`utils.parse_filename`](../src/utils.py)), and anything
off-pattern is **silently dropped**. The real corpus already breaks that parser:

| Real filename | Problem |
|---|---|
| `SAGAplus_V8K2_35_Gesture_iconic_deictic` | subtype contains `_` → split-based parser mis-segments |
| `MULTISIMO_S07_0007_gesture_move` vs `Ecolang_ad01_34_Gesture_ObjMan` | label case differs (`gesture`/`Gesture`) |
| `ZHUBO_9_070_26_Gesture_NA` | speaker token contains `_` → ambiguous boundaries |
| `clip_final_v2` | no label token → returns `None`, clip vanishes |

For a *released benchmark* this is not acceptable. Metadata must be **explicit, validated, and
standards-aligned**, not inferred.

---

## 1. Core principle — the manifest is the source of truth

> Filenames are **identifiers**, not metadata. All metadata lives in an explicit, validated
> **manifest** (`metadata.csv` + machine-readable sidecar), generated once and shipped with the dataset.

MaskBench may still *seed* the manifest by parsing filenames, but the parse result is **presented for
review/override** in the Builder (Source step) and frozen into the manifest — never trusted blindly at
release time.

---

## 2. Standards we align to

| Standard | Layer | Adopt |
|---|---|---|
| **FAIR principles** | Umbrella (Findable, Accessible, Interoperable, Reusable) | **now** |
| **Datasheets for Datasets** (Gebru et al. 2021) | Human-readable dataset documentation | **now** (ship `DATASHEET.md`) |
| **Croissant** (MLCommons) | Machine-readable dataset metadata (JSON-LD) | **now** (ship `croissant.json`) |
| **schema.org/Dataset** + **DCAT** | Findability (Google Dataset Search, repositories) | now (subset, via Croissant) |
| **DataCite Metadata Schema** | Citation / DOI (DANS assigns the DOI) | **at archival** |
| **W3C PROV-O** | Provenance of derived landmarks | **now** (provenance block, §5) |
| **Dublin Core** | Basic descriptive metadata | now (title, creator, date, rights, identifier) |
| **ELAN / EAF** + **ISO 24617-2** | Source-corpus gesture annotation lineage (SaGA, ECOLANG, MULTISIMO are ELAN-annotated) | reference / link only |
| **GDPR + informed consent** | Human-subjects privacy (German + EU corpora) | **mandatory** (§7) |

Pragmatic stack: **Datasheet (humans) + Croissant (machines) + PROV provenance + DataCite at archival.**

---

## 3. Controlled vocabularies

Free text is not allowed in these fields. Canonical values, case-normalised:

- **corpus** — `Ecolang | MULTISIMO | SaGA | SAGAplus | ZHUBO | GESRES | TEDM3D` (extend by PR, not ad hoc).
- **label** — `Gesture | NoGesture | Move` (canonical case; `gesture`→`Gesture`, etc.).
- **subtype** — per-corpus enum or `NA`. Multi-token subtypes (`iconic_deictic`) are **single values**, not split.
- **language** — ISO 639-1 (`en`, `de`, `zh`).
- **setting** — `narration | group_discussion | direction_giving | clinical | presentation`.

Each corpus has a one-row entry in `corpora.csv` (language, setting, consent, licence) so clip rows stay thin.

---

## 4. Identifiers & per-clip manifest

**`clip_id`** is an **opaque, stable, unique** string (the released filename stem). It must not be *parsed*
to recover meaning — meaning comes from the manifest row.

`metadata.csv` — one row per clip. Required (✚) / recommended (○):

| Column | Type | Vocab | Notes |
|---|---|---|---|
| ✚ `clip_id` | string | — | unique, stable |
| ✚ `corpus` | enum | §3 | |
| ✚ `speaker_id` | string | — | **globally unique** (prefix with corpus), drives split grouping |
| ✚ `label` | enum | §3 | the target |
| ○ `subtype` | enum | §3 | `NA` if none |
| ✚ `n_frames` | int | — | landmark array length |
| ✚ `fps` | number | — | must be > 0 |
| ✚ `is_mirror` | bool | — | augmentation flag |
| ○ `original_clip_id` | string | — | required when `is_mirror=true` — links mirror→source for split grouping |
| ○ `language`,`setting` | enum | §3 | joinable from `corpora.csv` |
| ○ `source_sha256` | string | — | source-video hash (dedup + traceability) |

Rules: every `clip_id` unique; every `speaker_id` resolvable in exactly one split (§6); `is_mirror=true` ⇒
`original_clip_id` present and in the **same split**; `fps>0`; `label`/`subtype` ∈ vocab.

---

## 5. Data dictionary — landmark arrays (`.npz`)

One `.npz` per clip. Keys, shapes, semantics (must be documented in the Datasheet & Croissant):

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| `world_body_landmarks` | `(n_frames, 23, 4)` | float | upper-body, **world / metric 3D** `[x, y, z, visibility]` |
| `image_body_landmarks` | `(n_frames, 23, 2)` | float | body in **pixels** `[x, y]` |
| `world_left_hand_landmarks` / `..._right_...` | `(n_frames, 21, 3)` | float | hand world `[x, y, z]` |
| `image_left_hand_landmarks` / `..._right_...` | `(n_frames, 21, 2)` | float | hand pixels `[x, y]` |

Hard rules:
- **Coordinate space, units and axis convention stated explicitly** (world = metric, origin at hip-center per MediaPipe).
- **Missing keypoints / frames encoded as `NaN`** — never `0`, never dropped, never ragged. (Fixing the ragged-array
  drop in `save_npz` is a prerequisite — it currently *loses* whole clips; see the run report's 6 dropped clips.)
- **Fixed keypoint index schema** published as a sidecar `keypoints.json` (index → joint name, MediaPipe ordering).

---

## 6. Splits manifest

`splits.json`: `{ seed, grouping_key: "speaker_id", test_speaker_fraction, train: [speaker_id…], test: [speaker_id…] }`.
Guarantees: **speaker-independent** (no speaker in both); **mirror pairs co-located** (via `original_clip_id`);
all corpora present in both splits; class balance within tolerance. Reproducible from `seed`.

---

## 7. Provenance & governance (mandatory)

**Provenance (PROV-O `wasGeneratedBy`)** — ships in `provenance.json`:
`maskbench_commit`, `model` (name, **sha256**, source URL — pin it; the Dockerfile's `latest/` URLs are not
reproducible), `config_sha256`, `run_id`, `generated_at`, `generator`.

**Governance** — per corpus in `corpora.csv`: `consent_status`, `ethics_ref`, `licence` (e.g. `CC-BY-4.0`),
`gdpr_basis`, `deidentification_verified` (no face mesh / no audio). A clip from a non-approved corpus is
**excluded**, not warned.

---

## 8. Release bundle (what a participant downloads)

```
<challenge>_release/
├── DATASHEET.md            # Datasheets-for-Datasets
├── croissant.json          # MLCommons Croissant (machine-readable)
├── metadata.csv            # §4 per-clip manifest (source of truth)
├── corpora.csv             # §3/§7 per-corpus attributes + governance
├── keypoints.json          # §5 index → joint schema
├── splits.json             # §6
├── provenance.json         # §7 PROV-O
├── train/  *.npy|*.npz      # §5 arrays
├── test/   *.npy|*.npz      # labels withheld
└── README.md               # format + submission rules
```

---

## 9. Validation rules (Builder gate)

Block release (error) / flag (warn):

- **error** — duplicate `clip_id`; `label`/`subtype`/`corpus` outside vocab; `fps≤0`; `is_mirror` without `original_clip_id`;
  speaker in both splits; mirror split-straddle; npz shape ≠ spec / ragged / contains non-`NaN` sentinels; corpus not consent-approved.
- **warn** — clip below detection-coverage threshold; multi-person frames (single-person assumption); corpus thin in a split;
  `Move` class under-represented; `source_sha256` missing.

These mirror the QA step in the Challenge Builder prototype.
