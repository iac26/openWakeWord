import numpy as np
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
        assert np.all(np.abs(batch) <= 32767)
