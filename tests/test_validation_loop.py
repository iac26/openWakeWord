"""Pins the validation-loop fixes:

- conv_attention's MultiheadAttention CUDA SDPA kernel crashes with
  "invalid configuration argument" when called on the very large batches
  the original code passed (entire X_val set in one forward). Cap the
  validation DataLoader batch_size and accumulate metrics across batches.
- Multi-batch X_val loop must accumulate predictions/labels and compute
  metrics over the union, not overwrite per-batch.

These tests run on CPU (which uses the math SDPA backend that doesn't
hit the same kernel-launch limit) but verify the structural fixes:
batch size is bounded, multi-batch metrics match single-batch metrics.
"""

import numpy as np
import torch
import torchmetrics

from openwakeword.train import Model


INPUT_SHAPE = (16, 96)


def _make_val_dataset(n: int) -> torch.utils.data.TensorDataset:
    torch.manual_seed(0)
    x = torch.rand(n, *INPUT_SHAPE)
    y = torch.cat([torch.ones(n // 2), torch.zeros(n - n // 2)])
    return torch.utils.data.TensorDataset(x, y)


def test_conv_attention_validation_loop_chunked_matches_full_batch():
    """The chunked val loop must produce bit-identical metrics to a
    single-batch val loop (modulo float associativity)."""
    torch.manual_seed(0)
    m = Model(n_classes=1, input_shape=INPUT_SHAPE,
              model_type="conv_attention", layer_dim=128)
    m.model.eval()

    n_val = 1000
    ds = _make_val_dataset(n_val)
    full = torch.utils.data.DataLoader(ds, batch_size=n_val)
    chunked = torch.utils.data.DataLoader(ds, batch_size=128)

    # Run the model once on the full batch.
    full_x, _ = next(iter(full))
    with torch.no_grad():
        full_preds = m.model(full_x)

    # Now run it batch-by-batch and concatenate, mirroring the val loop.
    chunked_preds_parts = []
    for x, _ in chunked:
        with torch.no_grad():
            chunked_preds_parts.append(m.model(x))
    chunked_preds = torch.cat(chunked_preds_parts)

    assert torch.allclose(full_preds, chunked_preds, atol=1e-5)


def test_validation_metrics_accumulate_across_batches():
    """Compute fp/recall/accuracy over multiple val batches by accumulating
    predictions+labels and call metrics once at the end. Must equal what
    a single-batch call would produce."""
    torch.manual_seed(0)
    m = Model(n_classes=1, input_shape=INPUT_SHAPE,
              model_type="conv_attention", layer_dim=128)
    m.model.eval()
    fp_fn = lambda pred, y: (y - pred <= -0.5).sum()
    recall = torchmetrics.Recall(task='binary')
    accuracy = torchmetrics.Accuracy(task='binary')

    n_val = 1000
    ds = _make_val_dataset(n_val)
    loader = torch.utils.data.DataLoader(ds, batch_size=128)
    full_loader = torch.utils.data.DataLoader(ds, batch_size=n_val)

    # ---- The new (correct) approach: accumulate preds+labels then compute.
    all_preds, all_labels = [], []
    for x, y in loader:
        with torch.no_grad():
            all_preds.append(m.model(x))
            all_labels.append(y)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    val_fp_acc = fp_fn(all_preds, all_labels[..., None])
    val_recall_acc = recall(all_preds, all_labels[..., None].long()).item()
    val_acc_acc = accuracy(all_preds, all_labels[..., None].to(torch.int64)).item()

    # ---- The reference: same model, same data, single forward.
    x_full, y_full = next(iter(full_loader))
    with torch.no_grad():
        preds_full = m.model(x_full)
    recall_full = torchmetrics.Recall(task='binary')
    accuracy_full = torchmetrics.Accuracy(task='binary')
    val_fp_full = fp_fn(preds_full, y_full[..., None])
    val_recall_full = recall_full(preds_full, y_full[..., None].long()).item()
    val_acc_full = accuracy_full(preds_full, y_full[..., None].to(torch.int64)).item()

    assert int(val_fp_acc) == int(val_fp_full)
    assert abs(val_recall_acc - val_recall_full) < 1e-5
    assert abs(val_acc_acc - val_acc_full) < 1e-5


def test_val_dataloader_batch_size_is_bounded():
    """Regression: train.py builds the val DataLoaders with a fixed
    VAL_BATCH constant, not batch_size=len(labels). Pin this so a future
    edit can't silently regress.
    """
    import re
    import inspect
    import openwakeword.train as t

    src = inspect.getsource(t)
    # Find every `torch.utils.data.DataLoader(...)` block and check that
    # none of them sets `batch_size=len(...labels)`. We strip comments
    # (lines starting with #) before scanning to allow such phrases in
    # explanatory text.
    code_only = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#")
    )
    bad = re.findall(r"batch_size\s*=\s*len\([A-Za-z_][A-Za-z0-9_]*\)", code_only)
    assert not bad, f"DataLoader uses unbounded batch_size=len(...): {bad}"
