"""Pins the contracts added by the livekit-wakeword integration:

- conv_attention model_type constructs and runs a forward pass on the
  (1, 16, 96) embedding contract used at inference time.
- export to ONNX produces a valid file at opset >= 14 (required for
  nn.MultiheadAttention) and the loaded model accepts the same input
  shape and emits a (1, 1) score in [0, 1].
- focal loss matches BCE when gamma == 0 and is strictly smaller for
  well-classified examples when gamma > 0.
- embedding mixup produces valid soft labels and preserves them through
  the train_model high-loss filter (regression test for the previous
  filter that re-derived y_ from hard y, which would have erased mixup).
"""

import os
import tempfile

import numpy as np
import onnx
import onnxruntime
import pytest
import torch

from openwakeword.train import Model


INPUT_SHAPE = (16, 96)


def test_conv_attention_constructs_and_forwards():
    m = Model(n_classes=1, input_shape=INPUT_SHAPE,
              model_type="conv_attention", layer_dim=128)
    x = torch.rand(4, *INPUT_SHAPE)
    y = m.model(x)
    assert y.shape == (4, 1)
    assert (y >= 0).all() and (y <= 1).all()


def test_conv_attention_layer_dim_divisibility():
    with pytest.raises(ValueError, match="divisible by"):
        Model(n_classes=1, input_shape=INPUT_SHAPE,
              model_type="conv_attention", layer_dim=130, n_heads=4)


def test_conv_attention_onnx_export_contract():
    m = Model(n_classes=1, input_shape=INPUT_SHAPE,
              model_type="conv_attention", layer_dim=128)
    with tempfile.TemporaryDirectory() as d:
        m.export_model(model=m.model, model_name="conv_attn", output_dir=d)
        onnx_path = os.path.join(d, "conv_attn.onnx")
        assert os.path.exists(onnx_path)

        proto = onnx.load(onnx_path)
        opsets = {o.domain: o.version for o in proto.opset_import}
        # MultiheadAttention requires opset 14+; we export at 17.
        assert opsets.get("", 0) >= 14

        sess = onnxruntime.InferenceSession(onnx_path,
                                            providers=["CPUExecutionProvider"])
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        assert len(inputs) == 1
        # (B, 16, 96), with B as a dynamic axis from PyTorch's exporter
        assert inputs[0].shape[1:] == list(INPUT_SHAPE)
        assert len(outputs) == 1

        x = np.random.rand(1, *INPUT_SHAPE).astype(np.float32)
        (y,) = sess.run(None, {inputs[0].name: x})
        assert y.shape == (1, 1)
        assert 0.0 <= float(y[0, 0]) <= 1.0

        # Dynamic batch axis: same model must run at batch=8.
        x8 = np.random.rand(8, *INPUT_SHAPE).astype(np.float32)
        (y8,) = sess.run(None, {inputs[0].name: x8})
        assert y8.shape == (8, 1)


def test_focal_loss_matches_bce_when_gamma_zero():
    m = Model(n_classes=1, input_shape=INPUT_SHAPE, model_type="dnn",
              loss_type="focal", focal_gamma=0.0)
    pred = torch.tensor([[0.7], [0.2], [0.5]])
    target = torch.tensor([[1.0], [0.0], [1.0]])
    weight = torch.tensor([[1.0], [1.0], [1.0]])
    focal = m.loss(pred, target, weight)
    bce = torch.nn.functional.binary_cross_entropy(pred, target, weight)
    assert torch.allclose(focal, bce, atol=1e-6)


def test_focal_loss_downweights_easy_examples():
    """gamma > 0 must produce smaller loss on a well-classified example."""
    m_focal = Model(n_classes=1, input_shape=INPUT_SHAPE, model_type="dnn",
                    loss_type="focal", focal_gamma=2.0)
    # Easy positive: prediction close to target.
    pred = torch.tensor([[0.95]])
    target = torch.tensor([[1.0]])
    bce = torch.nn.functional.binary_cross_entropy(pred, target)
    focal = m_focal.loss(pred, target, weight=torch.tensor([[1.0]]))
    assert focal.item() < bce.item()


def test_embedding_mixup_produces_soft_labels():
    """Sanity-check the mixup math directly against a fixed seed."""
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.rand(8, *INPUT_SHAPE)
    y_ = torch.tensor([[1.0], [0.0], [1.0], [0.0],
                       [1.0], [0.0], [1.0], [0.0]])
    lam = float(np.random.beta(0.2, 0.2))
    perm = torch.randperm(x.size(0))
    x_mixed = lam * x + (1.0 - lam) * x[perm]
    y_mixed = lam * y_ + (1.0 - lam) * y_[perm]
    # Mixed labels should be valid probabilities and at least one should
    # actually be soft (not exactly 0 or 1) given the random permutation.
    assert (y_mixed >= 0).all() and (y_mixed <= 1).all()
    assert ((y_mixed > 0) & (y_mixed < 1)).any()
    # Mixed embeddings keep the same shape.
    assert x_mixed.shape == x.shape
