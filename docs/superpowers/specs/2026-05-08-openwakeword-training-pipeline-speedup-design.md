# openWakeWord Training Pipeline Speedup — Design

**Date:** 2026-05-08
**Status:** Approved (awaiting user review of written spec)
**Scope:** `--augment_clips` and `--train_model` stages of `openwakeword/train.py`
**Behaviour bar:** Statistically equivalent to current main. Same augmentation distribution, same `augment2` per-batch semantics, same SGD batch shapes and LR schedule. RNG draw sequence may differ across workers; per-clip distribution is identical.

## Problem

Running `examples/hey_ari.yml` (n_samples=50000) on a 48 GB cloud GPU (L40S) shows:

- `compute_features_from_generator` reports a ~20 minute ETA for the augment stage. The GPU sits at 0% utilization the whole time.
- The training stage runs but the GPU is also barely used; per-step time is dominated by data plumbing, not the model (~50K params).

Two distinct root causes, both purely scheduling problems (no model, no hyperparameter changes needed):

1. **Augment stage is single-threaded CPU-bound.** The per-clip loop in `data.py:augment_clips` (`torchaudio.load` → `create_fixed_size_clip` → `audiomentations.Compose`) runs serially on one core at ~12 ms/clip. The downstream GPU steps (`torch_audiomentations` augment2 over batches of 16, then ONNX speech_embedding) finish in microseconds and idle waiting for the next batch.

2. **Training-stage `DataLoader` silently duplicates batches.** `train.py:844-857` wraps `mmap_batch_generator` in an `IterableDataset` and uses `DataLoader(num_workers=n_cpus, prefetch_factor=16)`. PyTorch forks N copies of the generator. None of them shard by `worker_info`, so each worker reads from the same memmap starting at `data_counter=0`. The DataLoader round-robins, feeding the trainer N copies of every batch in close succession before the underlying offset advances. Net effect: per-epoch unique-batch count is reduced by ~Nx and the training trajectory diverges from upstream-intended.

Both fixes are scheduling-only. They preserve all SGD math, all augmentation distributions, all `augment2` per-batch semantics.

## Design

Land both fixes in a single PR.

### Fix 1: Training DataLoader — remove duplicate workers

**File:** `openwakeword/train.py:844-857`

The minimal fix is one parameter change: set `num_workers=0`. The `IterDataset` wrapper itself is not broken — only the multi-worker fork path is, because each worker keeps its own copy of `mmap_batch_generator` with its own `data_counter` starting at 0. With `num_workers=0` the DataLoader iterates the generator in the main process, and the default `collate_fn` still converts the `(np.ndarray, np.ndarray)` tuple from `mmap_batch_generator.__next__` into tensors so `train_model`'s `data[0].to(self.device)` continues to work unchanged.

```python
# Was
X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                       batch_size=None, num_workers=n_cpus, prefetch_factor=16)

# Becomes
X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                       batch_size=None, num_workers=0)
```

The unused `n_cpus` computation immediately above the DataLoader can also go.

**Effect:** each step sees a unique batch. Same SGD batch shapes, same LR schedule, same downstream code. Step throughput drops (no more N-way duplication padding the queue) but each step is real work. Training quality moves toward upstream-correct.

### Fix 2: `augment_clips` — pipeline the per-clip CPU loop

**File:** `openwakeword/data.py:augment_clips`

Replace the inner serial `for clip in batch:` loop with a `torch.utils.data.DataLoader` over a per-clip `Dataset`. Workers handle `torchaudio.load` + `create_fixed_size_clip` + `audiomentations.Compose` in parallel; the main thread runs `augment2` on GPU + reverb + downstream embed. CPU augment for batch N+1 overlaps GPU work for batch N.

```python
class _PerClipAugDataset(torch.utils.data.Dataset):
    def __init__(self, clip_paths, total_length, sr, aug_probs):
        self.clip_paths = clip_paths
        self.total_length = total_length
        self.sr = sr
        self.aug_probs = aug_probs
        self._compose = None  # built per-worker via worker_init_fn

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        clip_data, clip_sr = torchaudio.load(self.clip_paths[idx])
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
    ds._compose = audiomentations.Compose([
        audiomentations.SevenBandParametricEQ(min_gain_db=-6, max_gain_db=6,
            p=ds.aug_probs["SevenBandParametricEQ"]),
        audiomentations.TanhDistortion(min_distortion=0.0001, max_distortion=0.10,
            p=ds.aug_probs["TanhDistortion"]),
    ])
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)
```

Main loop:

```python
loader = torch.utils.data.DataLoader(
    _PerClipAugDataset(clip_paths, total_length, sr, augmentation_probabilities),
    batch_size=batch_size,
    num_workers=min(8, max(1, (os.cpu_count() or 4) - 2)),
    worker_init_fn=_augment_worker_init,
    collate_fn=lambda batch: torch.from_numpy(np.stack(batch, axis=0)),
    prefetch_factor=4,
    persistent_workers=False,
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
for batch in loader:
    augmented_batch = augment2(samples=batch.unsqueeze(1).to(device),
                               sample_rate=sr).squeeze(1)
    if augmentation_probabilities["RIR"] >= np.random.random() and RIR_paths:
        rir_waveform, _ = torchaudio.load(random.choice(RIR_paths))
        augmented_batch = reverberate(augmented_batch.cpu(), rir_waveform, rescale_amp="avg")
    yield (augmented_batch.cpu().numpy() * 32767).astype(np.int16)
```

Worker count default: `min(8, cpu_count() - 2)`. Bench (200 clips, 20-core box): n=8 → 2.93x raw, n=20 → 2.17x (oversaturation). With pipeline overlap on top, expect 4–5x end-to-end on the augment stage.

**Equivalence argument:** each clip's augmentations come from independent `Compose` objects with independent RNG streams (seeded per worker). The serial loop also draws each clip's parameters independently. Per-clip output distribution is identical; the sequence of which clip is drawn first differs across runs. `augment2` still receives batches of `augmentation_batch_size=16`, so its `mode="per_batch"` semantics (one set of params per batch) are preserved exactly. Reverb still applies per-batch with the same per-batch coin flip.

## Risks & Mitigations

- **CUDA fork issue**: Linux DataLoader workers default to `fork` start method. If a worker accidentally imports a CUDA-touching module we get `RuntimeError: Cannot re-initialize CUDA in forked subprocess`. Mitigation: workers only call `torchaudio.load` + `audiomentations` (CPU-only). If it bites, set `multiprocessing_context="spawn"` (one-time startup cost per stage).
- **Reverb still serial**: kept in main loop deliberately — it's not the bottleneck, and parallelizing per-batch RIR convolution is trivial follow-up if ever needed.
- **`persistent_workers=False`**: workers tear down at end of each generator. We pay startup once per of the four `compute_features_from_generator` calls. With `num_workers=8` and `fork`, startup is sub-second; not worth the complexity of persistent workers across separate generator lifetimes.
- **Worker memory**: 8 × `audiomentations.Compose` is tens of MB — negligible.

## Verification Plan

Before/after measurements:

1. **Smoke test** — `training_workspace_local/heyari_small.yml` (1000 samples, GPU): run end-to-end on both branches, compare final `val_recall`, `val_n_fp`, `val_fp_per_hr` from `Model.history`. Expect equal-to-better metrics.
2. **Augment stage timing** — log wall-clock for each `compute_features_from_generator` call. Expect ~3–5x faster.
3. **Training stage timing** — log batches/sec. With Fix 1, batches/sec drops vs current (no more duplicates) but each step carries unique data; total wall-clock for `steps=N` should be similar or slightly higher with strictly better quality.
4. **Determinism check** — fix `torch.manual_seed`; two runs of the small smoke produce val metrics within sampling noise of each other.

Run the smoke test on the local 4 GB GPU before pushing to the cloud.

## Out of Scope

- Increasing `augmentation_batch_size` (rejected: `mode="per_batch"` reduces augmentation diversity at larger batches — true behavioural change).
- Increasing `batch_n_per_class` (rejected: alters effective LR and gradient noise — true SGD-dynamics change).
- `pin_memory=True` / `non_blocking=True` (marginal at 50K-param model size).
- Persistent worker pool across the four generator calls.
- Parallelizing reverb.

## Related Behavioural Drift Already in Main (audited)

Three pre-existing changes vs upstream `368c037`, all within "statistically equivalent" bar — left as-is:

1. `f()` reshape vectorization (commit `8c7795c`): not bit-identical to upstream. Upstream `range(0, x.shape[0]-n, n)` dropped the last window (62 of 64 at N=1024); new `x[:n_full].reshape(...)` keeps all of them. Strict correctness improvement (~1.6% more data per step), in the upstream-intended direction.
2. `_colored_noise` replaces `acoustics.generator.noise` (no Py3.12 wheels): same 1/f^α distribution, different RNG implementation. Distribution-equivalent.
3. `tts_batch_size: 256` (was 50): scheduling-only for Piper. Per-utterance noise/duration sampling is independent of batch size in Piper. Output distribution unchanged.

These do not block the planned fixes and require no rework.
