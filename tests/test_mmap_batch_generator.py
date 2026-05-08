import os
import tempfile
import numpy as np

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
