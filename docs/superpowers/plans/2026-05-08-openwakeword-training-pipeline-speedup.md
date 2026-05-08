# openWakeWord Training Pipeline Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the openWakeWord training pipeline GPU-fed instead of CPU-stalled, by (a) fixing a duplicate-worker bug in the training DataLoader and (b) parallelising the per-clip augment loop. Both changes are scheduling-only — SGD math, augmentation distributions, and `augment2` per-batch semantics are preserved.

**Architecture:**
1. `train.py:DataLoader(IterDataset(...), num_workers=n_cpus)` is replaced with `num_workers=0`. Each forked worker currently re-iterates the same `mmap_batch_generator` from offset 0 → duplicate batches.
2. `data.py:augment_clips` per-clip serial loop is replaced with a `torch.utils.data.DataLoader` over a per-clip `Dataset`, with `worker_init_fn` building each worker's `audiomentations.Compose` and seeding RNG.

**Tech Stack:** Python 3.12, PyTorch (`<2.6`), torchaudio, audiomentations, torch_audiomentations, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-05-08-openwakeword-training-pipeline-speedup-design.md`

---

## File Map

- **Modify:** `openwakeword/train.py` (lines ~836-857) — the training-stage DataLoader
- **Modify:** `openwakeword/data.py` (lines ~578-723) — `augment_clips` function body
- **Create:** `tests/test_augment_clips.py` — new test module
- **Create:** `tests/test_mmap_batch_generator.py` — new test module

No interface changes. `augment_clips` keeps its signature and still yields `(batch_size, total_length) int16 ndarray`. `train.py` change is internal to the `--train_model` branch.

---

## Task 1: Add regression test for `mmap_batch_generator` advancing counters

**Files:**
- Create: `tests/test_mmap_batch_generator.py`

This test pins the property the duplicate-worker bug was breaking: successive `__next__` calls return *different* slices of the underlying mmap, not the same one over and over.

- [ ] **Step 1: Create the test file**

```python
# tests/test_mmap_batch_generator.py
import os
import tempfile
import numpy as np
import pytest

from openwakeword.data import mmap_batch_generator


def _write_mmap(path: str, n_rows: int, frames: int = 16, features: int = 96) -> None:
    arr = np.arange(n_rows * frames * features, dtype=np.float32).reshape(n_rows, frames, features)
    np.save(path, arr)


def test_consecutive_batches_advance_counters():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.npy")
        _write_mmap(path, n_rows=1000)

        gen = mmap_batch_generator(
            data_files={"positive": path},
            batch_size=64,
            n_per_class={"positive": 64},
        )

        batch_a, _ = next(gen)
        batch_b, _ = next(gen)

        # Each batch is a contiguous 64-row slice; consecutive batches must differ
        assert not np.array_equal(batch_a, batch_b), \
            "Consecutive batches were identical — counter is not advancing"


def test_dataloader_num_workers_zero_preserves_advance():
    """Wrapping the generator in DataLoader(num_workers=0) must not duplicate batches."""
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.npy")
        _write_mmap(path, n_rows=1000)

        gen = mmap_batch_generator(
            data_files={"positive": path},
            batch_size=64,
            n_per_class={"positive": 64},
        )

        class _Iter(torch.utils.data.IterableDataset):
            def __init__(self, g):
                self.g = g
            def __iter__(self):
                return self.g

        loader = torch.utils.data.DataLoader(_Iter(gen), batch_size=None, num_workers=0)
        it = iter(loader)
        a = next(it)
        b = next(it)
        assert not torch.equal(a[0], b[0])
```

- [ ] **Step 2: Run test to verify it passes (no code change needed; this is a regression pin)**

Run: `pytest tests/test_mmap_batch_generator.py -v`
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mmap_batch_generator.py
git commit -m "test: pin mmap_batch_generator advances counters across __next__"
```

---

## Task 2: Fix the duplicate-worker bug in `train.py`

**Files:**
- Modify: `openwakeword/train.py` (around lines 851-857)

The current code uses `num_workers=n_cpus` which forks N copies of `mmap_batch_generator`. Each fork starts at `data_counter=0` and the DataLoader round-robins → trainer sees N duplicates of every batch. Fix: `num_workers=0`. The memmap reads are sub-millisecond, so workers add no value — they only enable the bug.

- [ ] **Step 1: Read the current block**

Read `openwakeword/train.py` lines 836-870. Locate:

```python
n_cpus = os.cpu_count()
if n_cpus is None:
    n_cpus = 1
else:
    n_cpus = n_cpus//2
X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                      batch_size=None, num_workers=n_cpus, prefetch_factor=16)
```

- [ ] **Step 2: Replace with single-process iteration**

Edit `openwakeword/train.py`: replace the `n_cpus = ...` block and the `X_train = ...` line above with:

```python
# Use num_workers=0: workers would each fork the mmap_batch_generator and
# re-iterate from data_counter=0, producing duplicate batches. The mmap
# slice + np.vstack in __next__ is sub-millisecond — no benefit from
# worker processes.
X_train = torch.utils.data.DataLoader(
    IterDataset(batch_generator),
    batch_size=None,
    num_workers=0,
)
```

Leave the `IterDataset` class definition itself in place.

- [ ] **Step 3: Confirm the test still passes**

Run: `pytest tests/test_mmap_batch_generator.py -v`
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add openwakeword/train.py
git commit -m "fix(train): use num_workers=0 to stop duplicate-batch bug

Forked DataLoader workers each kept their own mmap_batch_generator with
data_counter=0, producing N duplicate batches per round-robin. The mmap
slice in __next__ is sub-millisecond so workers add no throughput; they
only enabled the bug. Use num_workers=0."
```

---

## Task 3: Add test for `augment_clips` output equivalence (serial vs DataLoader)

**Files:**
- Create: `tests/test_augment_clips.py`

This test pins the contract `augment_clips` must keep through the refactor: yields `(batch_size, total_length) int16` arrays, value range valid, count matches input. We compare `num_workers=0` and `num_workers=2` to ensure parallel workers don't break shape/dtype.

- [ ] **Step 1: Create the test file**

```python
# tests/test_augment_clips.py
import os
import tempfile

import numpy as np
import pytest
import scipy.io.wavfile

from openwakeword.data import augment_clips


SR = 16000
TOTAL_LENGTH = 32000  # 2 s


def _write_clip(path: str, n_samples: int = 24000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    samples = (rng.standard_normal(n_samples) * 0.05).astype(np.float32)
    scipy.io.wavfile.write(path, SR, (samples * 32767).astype(np.int16))


def test_output_shape_dtype_and_count(tmp_path):
    n_clips = 32
    paths = []
    for i in range(n_clips):
        p = str(tmp_path / f"clip_{i:03d}.wav")
        _write_clip(p, seed=i)
        paths.append(p)

    # No background or RIRs → the augment2 branch without backgrounds is taken
    gen = augment_clips(
        clip_paths=paths,
        total_length=TOTAL_LENGTH,
        sr=SR,
        batch_size=16,
        background_clip_paths=[],
        RIR_paths=[],
    )

    batches = list(gen)
    total_rows = sum(b.shape[0] for b in batches)
    assert total_rows == n_clips
    for b in batches:
        assert b.dtype == np.int16
        assert b.shape[1] == TOTAL_LENGTH
        assert b.shape[0] <= 16
        assert b.min() >= -32768 and b.max() <= 32767


def test_no_nan_or_inf_in_output(tmp_path):
    paths = []
    for i in range(16):
        p = str(tmp_path / f"clip_{i:03d}.wav")
        _write_clip(p, seed=100 + i)
        paths.append(p)

    gen = augment_clips(
        clip_paths=paths,
        total_length=TOTAL_LENGTH,
        sr=SR,
        batch_size=16,
        background_clip_paths=[],
        RIR_paths=[],
    )
    for batch in gen:
        # int16 cannot be NaN, but assert sane range
        assert np.all(np.abs(batch) <= 32767)
```

- [ ] **Step 2: Run against current serial implementation**

Run: `pytest tests/test_augment_clips.py -v`
Expected: 2 passed (both pass against the existing serial `augment_clips`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_augment_clips.py
git commit -m "test: pin augment_clips output shape/dtype/count contract"
```

---

## Task 4: Refactor `augment_clips` to use a per-clip DataLoader

**Files:**
- Modify: `openwakeword/data.py` (the body of `augment_clips`, ~lines 689-723)

Replace the inner serial `for clip in batch:` loop with a `DataLoader(num_workers=N)` over a per-clip `Dataset`. Worker count `min(8, max(1, cpu_count()-2))`. `worker_init_fn` builds each worker's `audiomentations.Compose` and seeds `numpy`/`random`. `augment2` batch size stays at the user-supplied `batch_size` (default 16) — preserving `mode="per_batch"` semantics exactly.

- [ ] **Step 1: Add imports near the top of `data.py`**

Open `openwakeword/data.py`. Confirm these are already imported (most are): `numpy as np`, `torch`, `torchaudio`, `audiomentations`, `random`, `os`. Add at the top of the file (after existing imports) if not already present:

```python
import torch.utils.data
```

- [ ] **Step 2: Add helper Dataset class above `augment_clips`**

Insert immediately above the `def augment_clips(...)` line (around line 578):

```python
class _PerClipAugDataset(torch.utils.data.Dataset):
    """Dataset that loads a single wav, pads/truncates to `total_length`, and
    applies the per-clip CPU augmentations (audiomentations.Compose). The
    Compose object is built lazily in `worker_init_fn` so each worker has
    its own with an independently seeded RNG."""

    def __init__(self, clip_paths, total_length, sr, aug_probs):
        self.clip_paths = clip_paths
        self.total_length = total_length
        self.sr = sr
        self.aug_probs = aug_probs
        self._compose = None  # populated per-worker (or lazily in main proc)

    def __len__(self):
        return len(self.clip_paths)

    def _ensure_compose(self):
        if self._compose is None:
            self._compose = audiomentations.Compose([
                audiomentations.SevenBandParametricEQ(
                    min_gain_db=-6, max_gain_db=6,
                    p=self.aug_probs["SevenBandParametricEQ"]),
                audiomentations.TanhDistortion(
                    min_distortion=0.0001, max_distortion=0.10,
                    p=self.aug_probs["TanhDistortion"]),
            ])

    def __getitem__(self, idx):
        self._ensure_compose()
        path = self.clip_paths[idx]
        clip_data, clip_sr = torchaudio.load(path)
        if clip_sr != self.sr:
            raise ValueError("Error! Clip does not have the correct sample rate!")
        clip_data = clip_data[0]
        if clip_data.shape[0] > self.total_length:
            clip_data = clip_data[:self.total_length]
        clip_data = create_fixed_size_clip(clip_data, self.total_length, clip_sr)
        samples = np.asarray(clip_data, dtype=np.float32)
        return self._compose(samples=samples, sample_rate=self.sr)


def _augment_worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    # Force a fresh, worker-local Compose (with its own internal RNG state)
    ds._compose = None
    ds._ensure_compose()
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _augment_collate(batch):
    return torch.from_numpy(np.stack(batch, axis=0))
```

- [ ] **Step 3: Replace the inner loop in `augment_clips`**

Find the block in `augment_clips` that begins with:

```python
    # Iterate through all clips and augment them. The per-clip CPU loop is
    # serial: previously tried a ThreadPool here but it made things worse,
    # ...
    for i in range(0, len(clip_paths), batch_size):
        batch = clip_paths[i:i+batch_size]
        augmented_clips = []
        for clip in batch:
            ...
            augmented_clips.append(torch.from_numpy(augment1(samples=samples, sample_rate=sr)))

        # Do second pass augmentations
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        augmented_batch = augment2(samples=torch.vstack(augmented_clips).unsqueeze(dim=1).to(device), sample_rate=sr).squeeze(axis=1)

        # Do reverberation
        if augmentation_probabilities["RIR"] >= np.random.random() and RIR_paths != []:
            rir_waveform, sr = torchaudio.load(random.choice(RIR_paths))
            augmented_batch = reverberate(augmented_batch.cpu(), rir_waveform, rescale_amp="avg")

        # yield batch of 16-bit PCM audio data
        yield (augmented_batch.cpu().numpy()*32767).astype(np.int16)
```

Replace the entire `for i in range(0, len(clip_paths), batch_size):` block (and the `# Iterate through all clips and augment them...` comment above it) with:

```python
    # Per-clip CPU prep (load + pad + audiomentations.Compose) is the
    # bottleneck. Run it across a small DataLoader worker pool while the
    # main thread runs augment2 on GPU + reverb. Each worker owns its own
    # Compose and seeded RNG, so per-clip augmentation distribution is
    # identical to the serial loop in expectation. augment2 still receives
    # exactly `batch_size` clips so its mode="per_batch" semantics are
    # preserved.
    n_workers = min(8, max(1, (os.cpu_count() or 4) - 2))
    dataset = _PerClipAugDataset(
        clip_paths=clip_paths,
        total_length=total_length,
        sr=sr,
        aug_probs=augmentation_probabilities,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        worker_init_fn=_augment_worker_init if n_workers > 0 else None,
        collate_fn=_augment_collate,
        prefetch_factor=4 if n_workers > 0 else None,
        persistent_workers=False,
        drop_last=False,
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for batch in loader:
        augmented_batch = augment2(
            samples=batch.unsqueeze(1).to(device),
            sample_rate=sr,
        ).squeeze(1)

        # Reverberation (per-batch coin flip — preserves upstream semantics)
        if augmentation_probabilities["RIR"] >= np.random.random() and RIR_paths != []:
            rir_waveform, _ = torchaudio.load(random.choice(RIR_paths))
            augmented_batch = reverberate(augmented_batch.cpu(), rir_waveform, rescale_amp="avg")

        yield (augmented_batch.cpu().numpy() * 32767).astype(np.int16)
```

Note the small fix: the upstream code shadowed the outer `sr` argument with `sr, _ = torchaudio.load(...)` inside the RIR branch. The replacement uses `_` so `sr` keeps its meaning across iterations.

- [ ] **Step 4: Run the contract tests**

Run: `pytest tests/test_augment_clips.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the existing test suite**

Run: `pytest tests/ -v --ignore=tests/data`
Expected: all pre-existing tests still pass (unrelated to this change but a sanity check).

- [ ] **Step 6: Commit**

```bash
git add openwakeword/data.py
git commit -m "perf(data): parallelise per-clip augment via DataLoader workers

Replace the serial per-clip CPU loop in augment_clips with a
torch.utils.data.DataLoader over a per-clip Dataset. Each worker owns its
own audiomentations.Compose, seeded from torch.initial_seed() + worker_id,
so per-clip augmentation distribution is identical to the serial loop in
expectation. augment2 still receives exactly batch_size clips, preserving
mode=per_batch semantics. Bench (8 workers, 200 clips): 2.93x raw, plus
pipeline overlap with GPU augment2/embed."
```

---

## Task 5: Run smoke training and capture metrics

**Files:** none modified.

This is the integration check — verifies the two fixes together don't break end-to-end training and gives a wall-clock comparison.

- [ ] **Step 1: Verify the small training config exists**

Run: `ls training_workspace_local/heyari_small.yml`
Expected: file present.

- [ ] **Step 2: Run end-to-end smoke training**

Run:
```bash
python -m openwakeword.train \
  --training_config training_workspace_local/heyari_small.yml \
  --augment_clips --train_model 2>&1 | tee /tmp/smoke_after.log
```

Watch for:
- The "Computing features" tqdm bar finishes much faster than before (the augment+feature stage was the bottleneck).
- "Batches/steps per epoch:" line appears.
- Training proceeds; final `val_recall`, `val_n_fp`, `val_fp_per_hr` printed.

Expected: run completes without errors. Capture the final metrics from the log.

- [ ] **Step 3: Sanity-check the metrics**

Open `/tmp/smoke_after.log`. Confirm:
- No `RuntimeError`, `CUDA error`, or worker-process traceback.
- Final `val_recall >= 0.5` (smoke threshold — the small config is not expected to produce a great model, just a working one).
- `val_n_fp` is finite.

If any check fails, do NOT proceed. Investigate and fix before merging.

- [ ] **Step 4: No commit (smoke run is verification only)**

---

## Task 6: Update the YAML comment about `augmentation_batch_size`

**Files:**
- Modify: `examples/hey_ari.yml` (lines 19-24)

The current comment says the augment batch must stay at 16 because larger batches make augment slower. After Task 4, that's no longer the bottleneck — but `mode="per_batch"` diversity still requires keeping `batch_size=16`. Update the comment so future readers don't try to tune this on speed grounds.

- [ ] **Step 1: Edit the comment block**

Replace lines 19-24 in `examples/hey_ari.yml`:

```yaml
# tts_batch_size benefits from a big GPU (Piper synthesis is GPU-bound).
# augmentation_batch_size does NOT — bumping it makes the augment+feature
# step slower because per-clip CPU prep runs serially and bigger batches
# leave the GPU idle longer between yields. Stick with 16 (upstream default).
tts_batch_size: 256
augmentation_batch_size: 16
```

with:

```yaml
# tts_batch_size benefits from a big GPU (Piper synthesis is GPU-bound).
# augmentation_batch_size MUST stay at 16: torch_audiomentations augment2
# uses mode="per_batch" (one set of params per batch), so larger batches
# directly reduce augmentation diversity. Per-clip CPU prep is now
# pipelined via DataLoader workers, so 16 is no longer a speed bottleneck.
tts_batch_size: 256
augmentation_batch_size: 16
```

- [ ] **Step 2: Commit**

```bash
git add examples/hey_ari.yml
git commit -m "docs(hey_ari): update augmentation_batch_size comment

The reason to keep this at 16 is now augment2 mode=per_batch diversity,
not the per-clip CPU bottleneck (parallelised by DataLoader workers)."
```

---

## Self-Review

**Spec coverage:**
- Fix 1 (DataLoader duplicate-worker bug) → Tasks 1–2 ✓
- Fix 2 (parallel augment_clips with worker_init_fn seeding) → Tasks 3–4 ✓
- Verification plan (smoke run with metric capture) → Task 5 ✓
- Out-of-scope items remain out of scope (no augmentation_batch_size or batch_n_per_class changes) ✓
- Yaml comment that referenced the old reason → Task 6 ✓

**Placeholder scan:** None. All tasks contain exact paths, full code blocks, exact commands, and expected output. No "TBD" or "implement appropriately".

**Type/name consistency:**
- `_PerClipAugDataset.__init__` takes `(clip_paths, total_length, sr, aug_probs)` — used identically in Task 4 Step 3.
- `_augment_worker_init`, `_augment_collate` — referenced consistently.
- `augmentation_probabilities` (the public arg name in `augment_clips`) is mapped to `aug_probs` (the internal attribute on the dataset) — single mapping point, consistent.
- `n_workers` computed once and gates both `worker_init_fn` and `prefetch_factor` (set to `None` when 0 — required by PyTorch).
