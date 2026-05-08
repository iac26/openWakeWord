import torch
from torch import optim, nn
import torchinfo
import torchmetrics
import copy
import os
import sys
import uuid
import numpy as np
import scipy
import collections
import argparse
import logging
from tqdm import tqdm
import yaml
from pathlib import Path
import openwakeword
from openwakeword.data import generate_adversarial_texts, augment_clips, mmap_batch_generator
from openwakeword.utils import compute_features_from_generator
from openwakeword.utils import AudioFeatures


# Base model class for an openwakeword model
class Model(nn.Module):
    def __init__(self, n_classes=1, input_shape=(16, 96), model_type="dnn",
                 layer_dim=128, n_blocks=1, seconds_per_example=None,
                 loss_type="bce", focal_gamma=2.0,
                 embedding_mixup=False, mixup_alpha=0.2,
                 label_smoothing=0.0, weight_decay=0.01,
                 n_heads=4, n_conv=2, n_attn=1):
        super().__init__()

        # Store inputs as attributes
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.seconds_per_example = seconds_per_example
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.best_models = []
        self.best_model_scores = []
        self.best_val_fp = 1000
        self.best_val_accuracy = 0
        self.best_val_recall = 0
        self.best_train_recall = 0
        self.model_type = model_type

        # Loss config
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma

        # Embedding mixup config
        self.embedding_mixup = embedding_mixup
        self.mixup_alpha = mixup_alpha

        # Label smoothing config: target labels become y*(1-eps) + 0.5*eps,
        # i.e. 1.0 -> 1-eps/2 and 0.0 -> eps/2. Reduces over-confidence.
        # Disabled by default (0.0). Stacks with focal loss.
        self.label_smoothing = label_smoothing

        # Define model (currently on fully-connected network supported)
        if model_type == "dnn":
            # self.model = nn.Sequential(
            #     nn.Flatten(),
            #     nn.Linear(input_shape[0]*input_shape[1], layer_dim),
            #     nn.LayerNorm(layer_dim),
            #     nn.ReLU(),
            #     nn.Linear(layer_dim, layer_dim),
            #     nn.LayerNorm(layer_dim),
            #     nn.ReLU(),
            #     nn.Linear(layer_dim, n_classes),
            #     nn.Sigmoid() if n_classes == 1 else nn.ReLU(),
            # )

            class FCNBlock(nn.Module):
                def __init__(self, layer_dim):
                    super().__init__()
                    self.fcn_layer = nn.Linear(layer_dim, layer_dim)
                    self.relu = nn.ReLU()
                    self.layer_norm = nn.LayerNorm(layer_dim)

                def forward(self, x):
                    return self.relu(self.layer_norm(self.fcn_layer(x)))

            class Net(nn.Module):
                def __init__(self, input_shape, layer_dim, n_blocks=1, n_classes=1):
                    super().__init__()
                    self.flatten = nn.Flatten()
                    self.layer1 = nn.Linear(input_shape[0]*input_shape[1], layer_dim)
                    self.relu1 = nn.ReLU()
                    self.layernorm1 = nn.LayerNorm(layer_dim)
                    self.blocks = nn.ModuleList([FCNBlock(layer_dim) for i in range(n_blocks)])
                    self.last_layer = nn.Linear(layer_dim, n_classes)
                    self.last_act = nn.Sigmoid() if n_classes == 1 else nn.ReLU()

                def forward(self, x):
                    x = self.relu1(self.layernorm1(self.layer1(self.flatten(x))))
                    for block in self.blocks:
                        x = block(x)
                    x = self.last_act(self.last_layer(x))
                    return x
            self.model = Net(input_shape, layer_dim, n_blocks=n_blocks, n_classes=n_classes)
        elif model_type == "rnn":
            class Net(nn.Module):
                def __init__(self, input_shape, n_classes=1):
                    super().__init__()
                    self.layer1 = nn.LSTM(input_shape[-1], 64, num_layers=2, bidirectional=True,
                                          batch_first=True, dropout=0.0)
                    self.layer2 = nn.Linear(64*2, n_classes)
                    self.layer3 = nn.Sigmoid() if n_classes == 1 else nn.ReLU()

                def forward(self, x):
                    out, h = self.layer1(x)
                    return self.layer3(self.layer2(out[:, -1]))
            self.model = Net(input_shape, n_classes)
        elif model_type == "conv_attention":
            # Conv1D blocks for local syllable transitions, multi-head
            # self-attention for long-range temporal structure, mean-pool over
            # time. Replaces the dnn head's Flatten(16x96)->Linear which
            # destroys the temporal structure of the embedding sequence.
            # Reference: livekit-wakeword (Apache-2.0).
            class ConvAttentionNet(nn.Module):
                def __init__(self, input_shape, hidden_dim=128, n_heads=4,
                             n_conv=2, n_attn=1, n_classes=1):
                    super().__init__()
                    if hidden_dim % n_heads != 0:
                        raise ValueError(
                            f"hidden_dim ({hidden_dim}) must be divisible by "
                            f"n_heads ({n_heads}) for MultiheadAttention"
                        )
                    self.input_proj = nn.Linear(input_shape[1], hidden_dim)
                    conv_layers = []
                    for _ in range(n_conv):
                        conv_layers += [
                            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                            nn.GELU(),
                            nn.BatchNorm1d(hidden_dim),
                        ]
                    self.conv = nn.Sequential(*conv_layers)
                    self.attn_layers = nn.ModuleList([
                        nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
                        for _ in range(n_attn)
                    ])
                    self.attn_norms = nn.ModuleList([
                        nn.LayerNorm(hidden_dim) for _ in range(n_attn)
                    ])
                    self.head = nn.Linear(hidden_dim, n_classes)
                    self.last_act = nn.Sigmoid() if n_classes == 1 else nn.ReLU()

                def forward(self, x):
                    # x: (B, T, D) where T=16, D=96
                    h = self.input_proj(x)         # (B, T, H)
                    h = h.transpose(1, 2)          # (B, H, T) for Conv1d
                    h = self.conv(h)               # (B, H, T)
                    h = h.transpose(1, 2)          # (B, T, H)
                    for attn, norm in zip(self.attn_layers, self.attn_norms):
                        attn_out, _ = attn(h, h, h, need_weights=False)
                        h = norm(h + attn_out)
                    h = h.mean(dim=1)              # (B, H)
                    return self.last_act(self.head(h))
            self.model = ConvAttentionNet(input_shape, hidden_dim=layer_dim,
                                          n_heads=n_heads, n_conv=n_conv,
                                          n_attn=n_attn, n_classes=n_classes)

        # Define metrics
        if n_classes == 1:
            self.fp = lambda pred, y: (y-pred <= -0.5).sum()
            self.recall = torchmetrics.Recall(task='binary')
            self.accuracy = torchmetrics.Accuracy(task='binary')
        else:
            def multiclass_fp(p, y, threshold=0.5):
                probs = torch.nn.functional.softmax(p, dim=1)
                neg_ndcs = y == 0
                fp = (probs[neg_ndcs].argmax(axis=1) != 0 & (probs[neg_ndcs].max(axis=1)[0] > threshold)).sum()
                return fp

            def positive_class_recall(p, y, negative_class_label=0, threshold=0.5):
                probs = torch.nn.functional.softmax(p, dim=1)
                pos_ndcs = y != 0
                rcll = (probs[pos_ndcs].argmax(axis=1) > 0
                        & (probs[pos_ndcs].max(axis=1)[0] >= threshold)).sum()/pos_ndcs.sum()
                return rcll

            def positive_class_accuracy(p, y, negative_class_label=0):
                probs = torch.nn.functional.softmax(p, dim=1)
                pos_preds = probs.argmax(axis=1) != negative_class_label
                acc = (probs[pos_preds].argmax(axis=1) == y[pos_preds]).sum()/pos_preds.sum()
                return acc

            self.fp = multiclass_fp
            self.acc = positive_class_accuracy
            self.recall = positive_class_recall

        self.n_fp = 0
        self.val_fp = 0

        # Define logging dict (in-memory)
        self.history = collections.defaultdict(list)

        # Define optimizer and loss
        if n_classes == 1 and loss_type == "focal":
            # Binary focal loss with the same (input, target, weight) signature
            # as binary_cross_entropy. Down-weights well-classified examples by
            # (1-p_t)**gamma; gamma=2 is the value from the original paper.
            gamma = self.focal_gamma

            def _focal(input, target, weight=None, eps=1e-7):
                p = input.clamp(min=eps, max=1.0 - eps)
                bce = -(target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p))
                p_t = target * p + (1.0 - target) * (1.0 - p)
                loss = (1.0 - p_t).pow(gamma) * bce
                if weight is not None:
                    loss = loss * weight
                return loss.mean()
            self.loss = _focal
        elif n_classes == 1:
            self.loss = torch.nn.functional.binary_cross_entropy
        else:
            self.loss = nn.functional.cross_entropy
        # AdamW (decoupled weight decay) matches livekit-wakeword's recipe and
        # generalizes slightly better than plain Adam in our regime. Default
        # weight_decay=0.01 is the standard AdamW value; pass weight_decay=0.0
        # if you need exact-Adam behaviour.
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=0.0001, weight_decay=weight_decay
        )

    def save_model(self, output_path):
        """
        Saves the weights of a trained Pytorch model
        """
        if self.n_classes == 1:
            torch.save(self.model, output_path)

    def export_to_onnx(self, output_path, class_mapping=""):
        obj = self
        # opset 17 is required for nn.MultiheadAttention used by the
        # conv_attention head; safe for the dnn / rnn heads as well.
        opset = 17
        # Dynamic batch axis lets downstream code run inference at any batch
        # size; the runtime in openwakeword.model only reads shape[1] (the
        # fixed 16-frame time axis), so this is backwards compatible.
        dyn = {"input": {0: "batch"}, class_mapping: {0: "batch"}}
        # Make simple model for export based on model structure
        if self.n_classes == 1:
            # Save ONNX model
            torch.onnx.export(self.model.to("cpu"), torch.rand(self.input_shape)[None, ], output_path,
                              input_names=["input"], output_names=[class_mapping],
                              dynamic_axes=dyn, opset_version=opset)

        elif self.n_classes >= 1:
            class M(nn.Module):
                def __init__(self):
                    super().__init__()

                    # Define model
                    self.model = obj.model.to("cpu")

                def forward(self, x):
                    return torch.nn.functional.softmax(self.model(x), dim=1)

            # Save ONNX model
            torch.onnx.export(M(), torch.rand(self.input_shape)[None, ], output_path,
                              input_names=["input"], output_names=[class_mapping],
                              dynamic_axes=dyn, opset_version=opset)

    def lr_warmup_cosine_decay(self,
                               global_step,
                               warmup_steps=0,
                               hold=0,
                               total_steps=0,
                               start_lr=0.0,
                               target_lr=1e-3
                               ):
        # Cosine decay
        learning_rate = 0.5 * target_lr * (1 + np.cos(np.pi * (global_step - warmup_steps - hold)
                                           / float(total_steps - warmup_steps - hold)))

        # Target LR * progress of warmup (=1 at the final warmup step)
        warmup_lr = target_lr * (global_step / warmup_steps)

        # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether
        # `global_step < warmup_steps` and we're still holding.
        # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
        if hold > 0:
            learning_rate = np.where(global_step > warmup_steps + hold,
                                     learning_rate, target_lr)

        learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def forward(self, x):
        return self.model(x)

    def summary(self):
        return torchinfo.summary(self.model, input_size=(1,) + self.input_shape, device='cpu')

    def average_models(self, models=None):
        """Averages the weights of the provided models together to make a new model"""

        if models is None:
            models = self.best_models

        # Clone a model from the list as the base for the averaged model
        averaged_model = copy.deepcopy(models[0])
        averaged_model_dict = averaged_model.state_dict()

        # Only average float buffers/parameters. Integer buffers (e.g.
        # BatchNorm's num_batches_tracked, which conv_attention introduces)
        # can't be divided in-place and aren't meaningful to average — keep
        # them from models[0].
        avg_keys = [k for k, v in averaged_model_dict.items()
                    if v.is_floating_point()]

        for key in avg_keys:
            averaged_model_dict[key] *= 0

        for model in models:
            model_dict = model.state_dict()
            for key in avg_keys:
                averaged_model_dict[key] += model_dict[key]

        for key in avg_keys:
            averaged_model_dict[key] /= len(models)

        # Load the averaged weights into the model
        averaged_model.load_state_dict(averaged_model_dict)

        return averaged_model

    def _select_best_model(self, false_positive_validate_data, val_set_hrs=11.3, max_fp_per_hour=0.5, min_recall=0.20):
        """
        Select the top model based on the false positive rate on the validation data

        Args:
            false_positive_validate_data (torch.DataLoader): A dataloader with validation data
            n (int): The number of models to select

        Returns:
            list: A list of the top n models
        """
        # Force every saved checkpoint into eval mode before we score it.
        # Deep-copies were taken during training while the model was in
        # train mode, so without this, model(x_val) below uses BatchNorm's
        # batch statistics from the val batch instead of running stats —
        # which gave wrong FP rates and tainted the selection.
        for model in self.best_models:
            model.eval()

        # Get false positive rates for each model. Single tqdm over batches:
        # the previous structure put tqdm on the inner model loop, which
        # restarted the bar for every batch and looked like an infinite
        # loop on large false_positive_validate_data sets.
        false_positive_rates = [0]*len(self.best_models)
        for batch in tqdm(false_positive_validate_data,
                          desc="Find best checkpoints by false positive rate"):
            x_val, y_val = batch[0].to(self.device), batch[1].to(self.device)
            for mdl_ndx, model in enumerate(self.best_models):
                with torch.no_grad():
                    val_ps = model(x_val)
                    false_positive_rates[mdl_ndx] = false_positive_rates[mdl_ndx] + self.fp(val_ps, y_val[..., None]).detach().cpu().numpy()
        false_positive_rates = [fp/val_set_hrs for fp in false_positive_rates]

        all_recalls = [s["val_recall"] for s in self.best_model_scores]
        candidate_model_ndx = [ndx for ndx, fp in enumerate(false_positive_rates) if fp <= max_fp_per_hour]
        candidate_model_recall = [all_recalls[ndx] for ndx in candidate_model_ndx]

        # If nothing meets the strict FP target, fall back to the
        # highest-recall checkpoint regardless of FP. This is far better
        # than returning None and letting auto_train use self.model (the
        # final-step model from seq 3), which has been trained into the
        # ground on the escalated negative weight and effectively never
        # fires on real positives.
        if not candidate_model_recall or max(candidate_model_recall) <= min_recall:
            if not candidate_model_recall:
                logging.warning(
                    f"No checkpoints met FP/hr <= {max_fp_per_hour}. Relaxing "
                    f"the constraint and picking highest-recall checkpoint "
                    f"from the full pool ({len(self.best_models)} candidates)."
                )
            else:
                logging.warning(
                    f"No FP-qualifying checkpoint cleared min_recall={min_recall}. "
                    f"Relaxing and picking highest-recall checkpoint from the "
                    f"full pool."
                )
            if not all_recalls:
                logging.error("No checkpoints recorded during training; cannot select a best model.")
                return None
            best_ndx = int(np.argmax(all_recalls))
        else:
            best_ndx = candidate_model_ndx[int(np.argmax(candidate_model_recall))]

        best_model = self.best_models[best_ndx]
        best_score = self.best_model_scores[best_ndx]
        # Force eval mode: best_models entries were deep-copied during training
        # (so they're still in train mode with active BN batch-stats updates).
        # Inference must use eval mode or BN/dropout misbehave.
        best_model.eval()
        logging.info(
            "Best model from training step %s of %d candidates: recall=%.4f FP/hr=%.4f",
            best_score['training_step_ndx'],
            len(self.best_models),
            float(np.asarray(all_recalls[best_ndx]).item()),
            float(np.asarray(false_positive_rates[best_ndx]).item()),
        )
        return best_model

    def _find_optimal_threshold(self, model, X_val, false_positive_val_data,
                                val_set_hrs, target_fpph, min_recall=0.5):
        """Post-hoc threshold sweep on validation data.

        Mirrors livekit-wakeword/training/metrics.py:find_best_threshold:
        scans thresholds in [0.01, 0.99] and picks the one that maximizes
        recall on positives while keeping FPPH on the long negative set
        below target_fpph. Falls back to the threshold with the best
        balanced accuracy if no threshold satisfies both constraints.

        Returns: (best_threshold, recall, fpph)
        """
        model.eval()
        with torch.no_grad():
            # Positive vs negative predictions from X_val (labels split).
            pos_preds_parts, neg_preds_parts = [], []
            for batch in tqdm(X_val, desc="Threshold sweep: collecting X_val preds"):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                p = model(x).squeeze(-1).detach().cpu().numpy()
                y_np = y.detach().cpu().numpy().astype(bool)
                pos_preds_parts.append(p[y_np])
                neg_preds_parts.append(p[~y_np])
            pos_preds = np.concatenate(pos_preds_parts) if pos_preds_parts else np.array([])

            # Long negative-only set drives the FPPH calc since it has the
            # known duration (val_set_hrs).
            fp_preds_parts = []
            for batch in tqdm(false_positive_val_data, desc="Threshold sweep: collecting FP-set preds"):
                x = batch[0].to(self.device)
                fp_preds_parts.append(model(x).squeeze(-1).detach().cpu().numpy())
            fp_preds = np.concatenate(fp_preds_parts) if fp_preds_parts else np.array([])

        thresholds = np.arange(0.01, 1.0, 0.01)
        best, best_fallback = None, None
        for t in thresholds:
            t_float = float(t)
            recall = float(np.mean(pos_preds >= t_float)) if pos_preds.size else 0.0
            if recall < min_recall:
                continue
            fpph = float(np.sum(fp_preds >= t_float) / val_set_hrs) if val_set_hrs > 0 else float("inf")
            tnr = float(np.mean(fp_preds < t_float)) if fp_preds.size else 0.0
            balanced_acc = (recall + tnr) / 2.0
            entry = (t_float, recall, fpph, balanced_acc)

            if fpph <= target_fpph:
                if best is None or recall > best[1]:
                    best = entry
            if best_fallback is None or balanced_acc > best_fallback[3]:
                best_fallback = entry

        chosen = best if best is not None else best_fallback
        if chosen is None:
            logging.warning(
                f"No threshold met min_recall={min_recall}; defaulting to 0.5"
            )
            return 0.5, 0.0, float("inf")
        t, recall, fpph, _ = chosen
        return t, recall, fpph

    def auto_train(self, X_train, X_val, false_positive_val_data, steps=50000, max_negative_weight=1000,
                   target_fp_per_hour=0.2, val_set_hrs=None):
        """A sequence of training steps that produce relatively strong models
        automatically, based on validation data and performance targets provided.
        After training merges the best checkpoints and returns a single model.

        val_set_hrs: total hours of audio represented by false_positive_val_data.
        Required for FP/hr to be meaningful — was previously hardcoded to 11.3
        (the upstream openWakeWord val set), which made FP/hr nonsensical for
        any other val source. Caller must pass this; we no longer default it.
        """
        if val_set_hrs is None:
            raise ValueError(
                "val_set_hrs is required (hours of audio in false_positive_val_data). "
                "Compute it from the source .npy: n_frames * 0.08 / 3600."
            )

        # Sequence 1: bulk training. Job is to converge the model into the
        # right basin; checkpoints from this phase are NOT collected for the
        # final-model pool because LR=1e-4 produces noisy weights that don't
        # average meaningfully with the cooler phases. Validation still runs
        # so escalation logic and metrics history work normally.
        logging.info("#"*50 + "\nStarting training sequence 1...\n" + "#"*50)
        lr = 0.0001
        weights = np.linspace(1, max_negative_weight, int(steps)).tolist()
        val_steps = np.linspace(steps-int(steps*0.25), steps, 20).astype(np.int64)
        self.train_model(
                    X=X_train,
                    X_val=X_val,
                    false_positive_val_data=false_positive_val_data,
                    max_steps=steps,
                    negative_weight_schedule=weights,
                    val_steps=val_steps, warmup_steps=steps//5,
                    hold_steps=steps//3, lr=lr, val_set_hrs=val_set_hrs,
                    save_checkpoints_to_pool=False)

        # Sequence 2
        logging.info("#"*50 + "\nStarting training sequence 2...\n" + "#"*50)
        lr = lr/10
        steps = steps/10

        # Adjust weights as needed based on false positive per hour performance from first sequence
        if self.best_val_fp > target_fp_per_hour:
            max_negative_weight = max_negative_weight*2
            logging.info("Increasing weight on negative examples to reduce false positives...")

        weights = np.linspace(1, max_negative_weight, int(steps)).tolist()
        val_steps = np.linspace(1, steps, 20).astype(np.int16)
        self.train_model(
                    X=X_train,
                    X_val=X_val,
                    false_positive_val_data=false_positive_val_data,
                    max_steps=steps,
                    negative_weight_schedule=weights,
                    val_steps=val_steps, warmup_steps=steps//5,
                    hold_steps=steps//3, lr=lr, val_set_hrs=val_set_hrs)

        # Sequence 3
        logging.info("#"*50 + "\nStarting training sequence 3...\n" + "#"*50)
        lr = lr/10

        # Adjust weights as needed based on false positive per hour performance from second sequence
        if self.best_val_fp > target_fp_per_hour:
            max_negative_weight = max_negative_weight*2
            logging.info("Increasing weight on negative examples to reduce false positives...")

        weights = np.linspace(1, max_negative_weight, int(steps)).tolist()
        val_steps = np.linspace(1, steps, 20).astype(np.int16)
        self.train_model(
                    X=X_train,
                    X_val=X_val,
                    false_positive_val_data=false_positive_val_data,
                    max_steps=steps,
                    negative_weight_schedule=weights,
                    val_steps=val_steps, warmup_steps=steps//5,
                    hold_steps=steps//3, lr=lr, val_set_hrs=val_set_hrs)

        # Pick a single best checkpoint instead of averaging weights across
        # the three LR sequences. For the conv_attention head (BatchNorm +
        # MultiheadAttention), averaging running stats and attention weights
        # from very different training stages (LR 1e-4, 1e-5, 1e-6) tended to
        # smear the model and produce a worse final artifact than any single
        # checkpoint. Highest-recall checkpoint with FP <= target wins; fall
        # back to the running self.model if nothing qualifies.
        logging.info("Selecting single best checkpoint by FP/recall...")
        combined_model = self._select_best_model(
            false_positive_validate_data=false_positive_val_data,
            val_set_hrs=val_set_hrs,
            max_fp_per_hour=target_fp_per_hour,
            min_recall=0.20,
        )
        if combined_model is None:
            logging.warning("No checkpoint met the FP/recall bar; falling back "
                            "to the final-step model.")
            combined_model = self.model

        # Report validation metrics for combined model. Accumulate preds+labels
        # across all val batches; computing the metric from only the last batch
        # (the previous behaviour) gave nonsense recall/accuracy on small final
        # batches. Use FRESH torchmetrics objects so accumulated training-time
        # state doesn't poison the final report. Combined_model is in eval
        # mode by now (forced in _select_best_model).
        combined_model.eval()
        recall_fn = torchmetrics.Recall(task='binary').to(self.device)
        accuracy_fn = torchmetrics.Accuracy(task='binary').to(self.device)
        with torch.no_grad():
            all_preds, all_labels = [], []
            for batch in tqdm(X_val, desc="Val (X_val) on selected model"):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                all_preds.append(combined_model(x))
                all_labels.append(y)
            val_ps = torch.cat(all_preds)
            y = torch.cat(all_labels)

            combined_model_recall = recall_fn(val_ps, y[..., None]).detach().cpu().numpy()
            combined_model_accuracy = accuracy_fn(val_ps, y[..., None].to(torch.int64)).detach().cpu().numpy()

            combined_model_fp = 0
            for batch in tqdm(false_positive_val_data, desc="Val (FP set) on selected model"):
                x_val, y_val = batch[0].to(self.device), batch[1].to(self.device)
                val_ps = combined_model(x_val)
                combined_model_fp = combined_model_fp + self.fp(val_ps, y_val[..., None])

            combined_model_fp_per_hr = (combined_model_fp/val_set_hrs).detach().cpu().numpy()

        logging.info(
            "Final Model Accuracy: %.4f | Recall: %.4f | FP/hr: %.4f",
            float(np.asarray(combined_model_accuracy).item()),
            float(np.asarray(combined_model_recall).item()),
            float(np.asarray(combined_model_fp_per_hr).item()),
        )
        sys.stderr.flush()

        # Post-hoc threshold optimization. Sweeps thresholds on the val set
        # and picks the one that maximizes recall while keeping FPPH below
        # target_fp_per_hour. The exported model itself is unchanged; this
        # tells you the *recommended* deployment threshold instead of the
        # default 0.5. Mirrors livekit-wakeword's _find_optimal_threshold.
        opt_t, opt_recall, opt_fpph = self._find_optimal_threshold(
            model=combined_model,
            X_val=X_val,
            false_positive_val_data=false_positive_val_data,
            val_set_hrs=val_set_hrs,
            target_fpph=target_fp_per_hour,
        )
        logging.info(
            "Optimal deployment threshold: %.2f | recall=%.4f | FP/hr=%.4f (target was %.2f)",
            float(opt_t), float(opt_recall), float(opt_fpph), float(target_fp_per_hour),
        )
        sys.stderr.flush()

        return combined_model

    def predict_on_features(self, features, model=None):
        """
        Predict on Tensors of openWakeWord features corresponding to single audio clips

        Args:
            features (torch.Tensor): A Tensor of openWakeWord features with shape (batch, features)
            model (torch.nn.Module): A Pytorch model to use for prediction (default None, which will use self.model)

        Returns:
            torch.Tensor: An array of predictions of shape (batch, prediction), where 0 is negative and 1 is positive
        """
        if len(features) < 3:
            features = features[None, ]

        features = features.to(self.device)
        predictions = []
        for x in tqdm(features, desc="Predicting on clips"):
            x = x[None, ]
            batch = []
            for i in range(0, x.shape[1]-16, 1):  # step size of 1 (80 ms)
                batch.append(x[:, i:i+16, :])
            batch = torch.vstack(batch)
            if model is None:
                preds = self.model(batch)
            else:
                preds = model(batch)
            predictions.append(preds.detach().cpu().numpy()[None, ])

        return np.vstack(predictions)

    def predict_on_clips(self, clips, model=None):
        """
        Predict on Tensors of 16-bit 16 khz audio data

        Args:
            clips (np.ndarray): A Numpy array of audio clips with shape (batch, samples)
            model (torch.nn.Module): A Pytorch model to use for prediction (default None, which will use self.model)

        Returns:
            np.ndarray: An array of predictions of shape (batch, prediction), where 0 is negative and 1 is positive
        """

        # Get features from clips
        F = AudioFeatures(device='cpu', ncpu=4)
        features = F.embed_clips(clips, batch_size=16)

        # Predict on features
        preds = self.predict_on_features(torch.from_numpy(features), model=model)

        return preds

    def export_model(self, model, model_name, output_dir, temperature=1.0):
        """Saves the trained openwakeword model to ONNX format.

        temperature > 1 sharpens the output distribution at inference time
        without retraining: s' = sigmoid(T * logit(s)). Useful when focal
        loss + mixup leave real-audio scores capped around ~0.55; T≈11 maps
        0.55 -> 0.9 while keeping the natural threshold at 0.5. T=1.0 (the
        default) is a no-op.
        """

        if self.n_classes != 1:
            raise ValueError("Exporting models with more than one class is currently not supported! "
                             "Use the `export_to_onnx` function instead.")

        model_to_save = copy.deepcopy(model).to("cpu")

        if temperature != 1.0:
            class TempScaled(nn.Module):
                def __init__(self, m, T, eps=1e-6):
                    super().__init__()
                    self.m = m
                    self.T = float(T)
                    self.eps = eps

                def forward(self, x):
                    s = self.m(x).clamp(self.eps, 1.0 - self.eps)
                    logit = torch.log(s) - torch.log(1.0 - s)
                    return torch.sigmoid(self.T * logit)
            model_to_save = TempScaled(model_to_save, temperature)
            logging.info(f"Applying inference temperature scaling T={temperature} to exported ONNX")

        # Save ONNX model. opset 17 is required for nn.MultiheadAttention used
        # by the conv_attention head; safe for the dnn / rnn heads as well.
        # Dynamic batch axis lets downstream code run inference at any batch
        # size; the runtime in openwakeword.model only reads shape[1] (the
        # fixed 16-frame time axis), so this is backwards compatible.
        logging.info(f"####\nSaving ONNX mode as '{os.path.join(output_dir, model_name + '.onnx')}'")
        torch.onnx.export(
            model_to_save,
            torch.rand(self.input_shape)[None, ],
            os.path.join(output_dir, model_name + ".onnx"),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

        return None

    def train_model(self, X, max_steps, warmup_steps, hold_steps, X_val=None,
                    false_positive_val_data=None, positive_test_clips=None,
                    negative_weight_schedule=[1],
                    val_steps=[250], lr=0.0001, val_set_hrs=1,
                    save_checkpoints_to_pool=True):
        # Move models and main class to target device
        self.to(self.device)
        self.model.to(self.device)

        # Train model
        accumulation_steps = 1
        accumulated_samples = 0
        accumulated_predictions = torch.Tensor([]).to(self.device)
        accumulated_labels = torch.Tensor([]).to(self.device)
        for step_ndx, data in tqdm(enumerate(X, 0), total=max_steps, desc="Training"):
            # get the inputs; data is a list of [inputs, labels]
            x, y = data[0].to(self.device), data[1].to(self.device)
            y_ = y[..., None].to(torch.float32)

            # Embedding mixup: linearly interpolate (x, y_) with a permutation
            # of itself. Operates on pre-computed (T, D) embedding sequences,
            # not waveforms. Soft labels are kept for the loss; we re-derive
            # hard `y` from a 0.5 threshold so the high-loss filter below
            # still works. n_classes==1 only.
            if self.embedding_mixup and self.n_classes == 1:
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                perm = torch.randperm(x.size(0), device=x.device)
                x = lam * x + (1.0 - lam) * x[perm]
                y_ = lam * y_ + (1.0 - lam) * y_[perm]
                y = (y_.squeeze(-1) >= 0.5).long()

            # Label smoothing: 1.0 -> 1 - eps/2, 0.0 -> eps/2. Reduces
            # over-confidence at training time. Applied after mixup so it
            # caps the extremes of whatever soft target mixup produces.
            # n_classes == 1 only.
            if self.label_smoothing > 0.0 and self.n_classes == 1:
                eps = self.label_smoothing
                y_ = y_ * (1.0 - eps) + 0.5 * eps

            # Update learning rates
            for g in self.optimizer.param_groups:
                g['lr'] = self.lr_warmup_cosine_decay(step_ndx, warmup_steps=warmup_steps, hold=hold_steps,
                                                      total_steps=max_steps, target_lr=lr)

            # Note: optimizer.zero_grad() now lives down in the
            # backward+step block, NOT every iteration. With proper
            # gradient accumulation we need .grad to persist across
            # sub-batches; zeroing here would wipe accumulated gradients.

            # Get predictions for batch
            predictions = self.model(x)

            # Construct batch with only samples that have high loss. Track
            # `y_` through the same mask rather than re-deriving it from hard
            # `y`, so soft labels from embedding mixup are preserved into the
            # loss. Without mixup, y_ == y[..., None].float() and behaviour
            # is identical to the upstream implementation.
            neg_mask = (y == 0) & (predictions.squeeze() >= 0.001)  # thresholds were chosen arbitrarily but work well
            pos_mask = (y == 1) & (predictions.squeeze() < 0.999)
            predictions = torch.cat((predictions[neg_mask], predictions[pos_mask]))
            y_ = torch.cat((y_[neg_mask], y_[pos_mask]))
            y = torch.cat((y[neg_mask], y[pos_mask]))

            # Set weights for batch
            if len(negative_weight_schedule) == 1:
                w = torch.ones(y.shape[0])*negative_weight_schedule[0]
                pos_ndcs = y == 1
                w[pos_ndcs] = 1
                w = w[..., None]
            else:
                if self.n_classes == 1:
                    w = torch.ones(y.shape[0])*negative_weight_schedule[step_ndx]
                    pos_ndcs = y == 1
                    w[pos_ndcs] = 1
                    w = w[..., None]

            if predictions.shape[0] != 0:
                # Gradient accumulation: when the high-loss filter shrinks the
                # batch below 128 samples, accumulate gradients across multiple
                # sub-batches before stepping.
                #
                # PRIOR BUG: backward only ran in the else branch, so gradients
                # from the smaller "accumulating" iterations were silently lost
                # — only the final sub-batch's gradient (pre-scaled by
                # 1/accumulation_steps) ever reached the optimizer. That
                # effectively skipped 2 out of every 3 updates in late-stage
                # training (where the filter rejects most negatives) and
                # under-trained phases 2 and 3.
                #
                # Correct pattern: backward each sub-batch unscaled (so .grad
                # holds the SUM of per-sub-batch gradients), then divide by
                # accumulation_steps before optimizer.step() so the applied
                # update corresponds to the MEAN gradient — same direction
                # and magnitude as if we'd seen one large batch. The previous
                # `loss = loss/accumulation_steps` pre-divide was a partial
                # attempt at this, but combined with the missed backwards it
                # produced harmonic-weighted gradient sums, not a true mean.
                loss = self.loss(predictions, y_ if self.n_classes == 1 else y, w.to(self.device))
                accumulated_samples += predictions.shape[0]

                if predictions.shape[0] >= 128:
                    accumulated_predictions = predictions
                    accumulated_labels = y_

                loss.backward()  # raw grads accumulate into .grad

                if accumulated_samples < 128:
                    accumulation_steps += 1
                    accumulated_predictions = torch.cat((accumulated_predictions, predictions))
                    accumulated_labels = torch.cat((accumulated_labels, y_))
                else:
                    if accumulation_steps > 1:
                        # Average the accumulated gradients across sub-batches
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad.data.div_(accumulation_steps)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accumulation_steps = 1
                    accumulated_samples = 0

                    self.history["loss"].append(loss.detach().cpu().numpy())

                    # Compute training metrics and log them. With embedding
                    # mixup the labels can be soft (in [0, 1]); torchmetrics
                    # binary classifiers reject anything other than {0, 1},
                    # so threshold for the metric only — the loss above
                    # already used the soft labels.
                    fp = self.fp(accumulated_predictions, accumulated_labels if self.n_classes == 1 else y)
                    self.n_fp += fp
                    metric_labels = (accumulated_labels >= 0.5).long() if self.n_classes == 1 else accumulated_labels
                    # torchmetrics.Recall is stateful: reset before each call
                    # so each batch's recall is independent. Without this,
                    # every reported recall is the cumulative running average
                    # across all calls — which silently poisoned every
                    # checkpoint's val_recall in best_model_scores and led
                    # _select_best_model to pick badly.
                    self.recall.reset()
                    self.history["recall"].append(self.recall(accumulated_predictions, metric_labels).detach().cpu().numpy())

                    accumulated_predictions = torch.Tensor([]).to(self.device)
                    accumulated_labels = torch.Tensor([]).to(self.device)

            # Validation block. Switch the model to eval mode for the entire
            # block — without this, BatchNorm uses the val-batch statistics
            # (which differ wildly from the train distribution because the
            # val set is 50:50 pos:neg while training batches are heavily
            # negative-weighted). That gave saved val_recall values up to
            # ~60x lower than the same model's recall when re-evaluated in
            # eval mode after _select_best_model. Restore train mode in a
            # finally so a mid-val exception can't leave us in eval mode
            # for subsequent training steps (which would freeze BN updates).
            is_val_step = step_ndx in val_steps and step_ndx > 1
            if is_val_step:
                self.model.eval()

            try:
                # Run validation and log validation metrics
                if is_val_step and false_positive_val_data is not None:
                    # Get false positives per hour with false positive data
                    val_fp = 0
                    for val_step_ndx, data in enumerate(false_positive_val_data):
                        with torch.no_grad():
                            x_val, y_val = data[0].to(self.device), data[1].to(self.device)
                            val_predictions = self.model(x_val)
                            val_fp += self.fp(val_predictions, y_val[..., None])
                    val_fp_per_hr = (val_fp/val_set_hrs).detach().cpu().numpy()
                    self.history["val_fp_per_hr"].append(val_fp_per_hr)

                # Get recall on test clips
                if is_val_step and positive_test_clips is not None:
                    tp = 0
                    fn = 0
                    for val_step_ndx, data in enumerate(positive_test_clips):
                        with torch.no_grad():
                            x_val = data[0].to(self.device)
                            batch = []
                            for i in range(0, x_val.shape[1]-16, 1):
                                batch.append(x_val[:, i:i+16, :])
                            batch = torch.vstack(batch)
                            preds = self.model(batch)
                            if any(preds >= 0.5):
                                tp += 1
                            else:
                                fn += 1
                    self.history["positive_test_clips_recall"].append(tp/(tp + fn))

                if is_val_step and X_val is not None:
                    # Concatenate predictions/labels across batches before
                    # computing metrics. The original code overwrote
                    # val_recall / val_acc / val_fp each iteration; that was
                    # harmless when X_val ran in a single huge batch, but
                    # breaks now that we cap the val batch size to keep
                    # MultiheadAttention's CUDA SDPA kernel from crashing.
                    all_preds = []
                    all_labels = []
                    for val_step_ndx, data in enumerate(X_val):
                        with torch.no_grad():
                            x_val, y_val = data[0].to(self.device), data[1].to(self.device)
                            all_preds.append(self.model(x_val))
                            all_labels.append(y_val)
                    all_preds = torch.cat(all_preds)
                    all_labels = torch.cat(all_labels)
                    # Reset stateful metrics so val_recall/val_acc reflect
                    # THIS val_step's predictions only, not the cumulative
                    # running average across every previous train-batch + val
                    # call.
                    self.recall.reset()
                    self.accuracy.reset()
                    val_recall = self.recall(all_preds, all_labels[..., None]).detach().cpu().numpy()
                    val_acc = self.accuracy(all_preds, all_labels[..., None].to(torch.int64))
                    val_fp = self.fp(all_preds, all_labels[..., None])
                    self.history["val_accuracy"].append(val_acc.detach().cpu().numpy())
                    self.history["val_recall"].append(val_recall)
                    self.history["val_n_fp"].append(val_fp.detach().cpu().numpy())
            finally:
                if is_val_step:
                    self.model.train()

            # Save models with a validation score above/below the 90th percentile
            # of the validation scores up to that point. Phase-1 (long, high-LR)
            # checkpoints are excluded from the pool by passing
            # save_checkpoints_to_pool=False from auto_train: phase 1's job is
            # to find the basin, not to produce deployable checkpoints.
            # Per Izmailov et al. 2018 (SWA), averaging is meaningful only
            # within the same LR regime late in training.
            if save_checkpoints_to_pool and step_ndx in val_steps and step_ndx > 1:
                if self.history["val_n_fp"][-1] <= np.percentile(self.history["val_n_fp"], 50) and \
                   self.history["val_recall"][-1] >= np.percentile(self.history["val_recall"], 5):
                    # logging.info("Saving checkpoint with metrics >= to targets!")
                    self.best_models.append(copy.deepcopy(self.model))
                    self.best_model_scores.append({"training_step_ndx": step_ndx, "val_n_fp": self.history["val_n_fp"][-1],
                                                   "val_recall": self.history["val_recall"][-1],
                                                   "val_accuracy": self.history["val_accuracy"][-1],
                                                   "val_fp_per_hr": self.history.get("val_fp_per_hr", [0])[-1]})
                    self.best_val_recall = self.history["val_recall"][-1]
                    self.best_val_accuracy = self.history["val_accuracy"][-1]

            if step_ndx == max_steps-1:
                break


if __name__ == '__main__':
    # Configure root logger explicitly: previously we relied on the default
    # WARNING-level logger, which let early INFO logs slip through somehow but
    # silently dropped late-stage INFO records (the "Best model from..." /
    # final-stats / threshold-sweep block disappeared even though the script
    # ran to completion and produced the ONNX). Force a known config + line
    # buffering on stderr.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
        force=True,
    )
    try:
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Get training config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_config",
        help="The path to the training config file (required)",
        type=str,
        required=True
    )
    parser.add_argument(
        "--generate_clips",
        help="Execute the synthetic data generation process",
        action="store_true",
        default="False",
        required=False
    )
    parser.add_argument(
        "--augment_clips",
        help="Execute the synthetic data augmentation process",
        action="store_true",
        default="False",
        required=False
    )
    parser.add_argument(
        "--overwrite",
        help="Overwrite existing openwakeword features when the --augment_clips flag is used",
        action="store_true",
        default="False",
        required=False
    )
    parser.add_argument(
        "--train_model",
        help="Execute the model training process",
        action="store_true",
        default="False",
        required=False
    )
    args = parser.parse_args()
    config = yaml.load(open(args.training_config, 'r').read(), yaml.Loader)

    # Define output locations
    config["output_dir"] = os.path.abspath(config["output_dir"])
    if not os.path.exists(config["output_dir"]):
        os.mkdir(config["output_dir"])
    if not os.path.exists(os.path.join(config["output_dir"], config["model_name"])):
        os.mkdir(os.path.join(config["output_dir"], config["model_name"]))

    positive_train_output_dir = os.path.join(config["output_dir"], config["model_name"], "positive_train")
    positive_test_output_dir = os.path.join(config["output_dir"], config["model_name"], "positive_test")
    negative_train_output_dir = os.path.join(config["output_dir"], config["model_name"], "negative_train")
    negative_test_output_dir = os.path.join(config["output_dir"], config["model_name"], "negative_test")
    feature_save_dir = os.path.join(config["output_dir"], config["model_name"])

    # RIR + background audio paths are only consumed during the
    # --augment_clips step. Resolved lazily inside that branch so users who
    # already have computed feature .npy files can run --train_model alone
    # without needing mit_rirs / audioset on disk.
    rir_paths = None
    background_paths = None

    if args.generate_clips is True:
        # Piper TTS is only needed for synthetic clip generation. Imported
        # lazily so --train_model alone (with .npy features already on disk)
        # doesn't require piper-sample-generator to be installed.
        sys.path.insert(0, os.path.abspath(config["piper_sample_generator_path"]))

        # PyTorch 2.6+ changed torch.load to default weights_only=True, which
        # rejects piper's checkpoint (it pickles custom SynthesizerTrn classes).
        # Patch the default back to False so generate_samples can load the model.
        _orig_torch_load = torch.load
        def _torch_load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_torch_load(*args, **kwargs)
        torch.load = _torch_load_compat  # type: ignore[assignment]

        try:
            from generate_samples import generate_samples  # legacy flat layout
        except ImportError:
            from piper_sample_generator.generate_samples import generate_samples  # current packaged layout

        # Generate positive clips for training
        logging.info("#"*50 + "\nGenerating positive clips for training\n" + "#"*50)
        if not os.path.exists(positive_train_output_dir):
            os.mkdir(positive_train_output_dir)
        n_current_samples = len(os.listdir(positive_train_output_dir))
        if n_current_samples <= 0.95*config["n_samples"]:
            generate_samples(
                text=config["target_phrase"], max_samples=config["n_samples"]-n_current_samples,
                batch_size=config["tts_batch_size"],
                noise_scales=[0.98], noise_scale_ws=[0.98], length_scales=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.33],
                output_dir=positive_train_output_dir, auto_reduce_batch_size=True,
                file_names=[uuid.uuid4().hex + ".wav" for i in range(config["n_samples"])]
            )
            torch.cuda.empty_cache()
        else:
            logging.warning(f"Skipping generation of positive clips for training, as ~{config['n_samples']} already exist")

        # Generate positive clips for testing
        logging.info("#"*50 + "\nGenerating positive clips for testing\n" + "#"*50)
        if not os.path.exists(positive_test_output_dir):
            os.mkdir(positive_test_output_dir)
        n_current_samples = len(os.listdir(positive_test_output_dir))
        if n_current_samples <= 0.95*config["n_samples_val"]:
            generate_samples(text=config["target_phrase"], max_samples=config["n_samples_val"]-n_current_samples,
                             batch_size=config["tts_batch_size"],
                             noise_scales=[1.0], noise_scale_ws=[1.0], length_scales=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.33],
                             output_dir=positive_test_output_dir, auto_reduce_batch_size=True)
            torch.cuda.empty_cache()
        else:
            logging.warning(f"Skipping generation of positive clips testing, as ~{config['n_samples_val']} already exist")

        # Generate adversarial negative clips for training
        logging.info("#"*50 + "\nGenerating negative clips for training\n" + "#"*50)
        if not os.path.exists(negative_train_output_dir):
            os.mkdir(negative_train_output_dir)
        n_current_samples = len(os.listdir(negative_train_output_dir))
        if n_current_samples <= 0.95*config["n_samples"]:
            adversarial_texts = config["custom_negative_phrases"]
            for target_phrase in config["target_phrase"]:
                adversarial_texts.extend(generate_adversarial_texts(
                    input_text=target_phrase,
                    N=config["n_samples"]//len(config["target_phrase"]),
                    include_partial_phrase=1.0,
                    # Default upstream value (0.2) keeps 20% of input words
                    # verbatim, producing adversarials like "hey <near-rhyme>"
                    # — phonetically *too* close to the positive cluster, which
                    # over-trains the model to reject anything resembling
                    # "hey ari" and causes under-firing on the real phrase.
                    # 0.0 forces both words to be replaced.
                    include_input_words=0.0))
            generate_samples(text=adversarial_texts, max_samples=config["n_samples"]-n_current_samples,
                             batch_size=config["tts_batch_size"]//3,
                             noise_scales=[0.98], noise_scale_ws=[0.98], length_scales=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.33],
                             output_dir=negative_train_output_dir, auto_reduce_batch_size=True,
                             file_names=[uuid.uuid4().hex + ".wav" for i in range(config["n_samples"])]
                             )
            torch.cuda.empty_cache()
        else:
            logging.warning(f"Skipping generation of negative clips for training, as ~{config['n_samples']} already exist")

        # Generate adversarial negative clips for testing
        logging.info("#"*50 + "\nGenerating negative clips for testing\n" + "#"*50)
        if not os.path.exists(negative_test_output_dir):
            os.mkdir(negative_test_output_dir)
        n_current_samples = len(os.listdir(negative_test_output_dir))
        if n_current_samples <= 0.95*config["n_samples_val"]:
            adversarial_texts = config["custom_negative_phrases"]
            for target_phrase in config["target_phrase"]:
                adversarial_texts.extend(generate_adversarial_texts(
                    input_text=target_phrase,
                    N=config["n_samples_val"]//len(config["target_phrase"]),
                    include_partial_phrase=1.0,
                    # Default upstream value (0.2) keeps 20% of input words
                    # verbatim, producing adversarials like "hey <near-rhyme>"
                    # — phonetically *too* close to the positive cluster, which
                    # over-trains the model to reject anything resembling
                    # "hey ari" and causes under-firing on the real phrase.
                    # 0.0 forces both words to be replaced.
                    include_input_words=0.0))
            generate_samples(text=adversarial_texts, max_samples=config["n_samples_val"]-n_current_samples,
                             batch_size=config["tts_batch_size"]//3,
                             noise_scales=[1.0], noise_scale_ws=[1.0], length_scales=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.33],
                             output_dir=negative_test_output_dir, auto_reduce_batch_size=True)
            torch.cuda.empty_cache()
        else:
            logging.warning(f"Skipping generation of negative clips for testing, as ~{config['n_samples_val']} already exist")

    # Do Data Augmentation
    if args.augment_clips is True:
        # Resolve RIR + background paths now (skipped at startup so that
        # --train_model alone doesn't require these directories on disk).
        rir_paths = [i.path for j in config["rir_paths"] for i in os.scandir(j)]
        background_paths = []
        if len(config["background_paths_duplication_rate"]) != len(config["background_paths"]):
            config["background_paths_duplication_rate"] = [1]*len(config["background_paths"])
        for background_path, duplication_rate in zip(config["background_paths"], config["background_paths_duplication_rate"]):
            background_paths.extend([i.path for i in os.scandir(background_path)]*duplication_rate)

        # Set the total length of the training clips based on the ~median
        # generated clip duration, rounded to the nearest 1000 samples; 32000
        # samples (2s) is treated as a default snap point. Only used by the
        # augment_clips pipeline below, so it's computed inside this branch.
        n = 50  # sample size
        positive_clips = [str(i) for i in Path(positive_test_output_dir).glob("*.wav")]
        duration_in_samples = []
        for i in range(n):
            sr, dat = scipy.io.wavfile.read(positive_clips[np.random.randint(0, len(positive_clips))])
            duration_in_samples.append(len(dat))

        config["total_length"] = int(round(np.median(duration_in_samples)/1000)*1000) + 12000  # add 750 ms to clip duration as buffer
        if config["total_length"] < 32000:
            config["total_length"] = 32000  # set a minimum of 32000 samples (2 seconds)
        elif abs(config["total_length"] - 32000) <= 4000:
            config["total_length"] = 32000

        if not os.path.exists(os.path.join(feature_save_dir, "positive_features_train.npy")) or args.overwrite is True:
            positive_clips_train = [str(i) for i in Path(positive_train_output_dir).glob("*.wav")]*config["augmentation_rounds"]
            positive_clips_train_generator = augment_clips(positive_clips_train, total_length=config["total_length"],
                                                           batch_size=config["augmentation_batch_size"],
                                                           background_clip_paths=background_paths,
                                                           RIR_paths=rir_paths)

            positive_clips_test = [str(i) for i in Path(positive_test_output_dir).glob("*.wav")]*config["augmentation_rounds"]
            positive_clips_test_generator = augment_clips(positive_clips_test, total_length=config["total_length"],
                                                          batch_size=config["augmentation_batch_size"],
                                                          background_clip_paths=background_paths,
                                                          RIR_paths=rir_paths)

            negative_clips_train = [str(i) for i in Path(negative_train_output_dir).glob("*.wav")]*config["augmentation_rounds"]
            negative_clips_train_generator = augment_clips(negative_clips_train, total_length=config["total_length"],
                                                           batch_size=config["augmentation_batch_size"],
                                                           background_clip_paths=background_paths,
                                                           RIR_paths=rir_paths)

            negative_clips_test = [str(i) for i in Path(negative_test_output_dir).glob("*.wav")]*config["augmentation_rounds"]
            negative_clips_test_generator = augment_clips(negative_clips_test, total_length=config["total_length"],
                                                          batch_size=config["augmentation_batch_size"],
                                                          background_clip_paths=background_paths,
                                                          RIR_paths=rir_paths)

            # Compute features and save to disk via memmapped arrays
            logging.info("#"*50 + "\nComputing openwakeword features for generated samples\n" + "#"*50)
            n_cpus = os.cpu_count()
            if n_cpus is None:
                n_cpus = 1
            else:
                n_cpus = n_cpus//2
            compute_features_from_generator(positive_clips_train_generator, n_total=len(os.listdir(positive_train_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=os.path.join(feature_save_dir, "positive_features_train.npy"),
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)

            compute_features_from_generator(negative_clips_train_generator, n_total=len(os.listdir(negative_train_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=os.path.join(feature_save_dir, "negative_features_train.npy"),
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)

            compute_features_from_generator(positive_clips_test_generator, n_total=len(os.listdir(positive_test_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=os.path.join(feature_save_dir, "positive_features_test.npy"),
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)

            compute_features_from_generator(negative_clips_test_generator, n_total=len(os.listdir(negative_test_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=os.path.join(feature_save_dir, "negative_features_test.npy"),
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)
        else:
            logging.warning("Openwakeword features already exist, skipping data augmentation and feature generation")

    # Create openwakeword model
    if args.train_model is True:
        input_shape = np.load(os.path.join(feature_save_dir, "positive_features_test.npy")).shape[1:]

        oww = Model(
            n_classes=1, input_shape=input_shape,
            model_type=config["model_type"],
            layer_dim=config["layer_size"],
            seconds_per_example=1280*input_shape[0]/16000,
            loss_type=config.get("loss_type", "bce"),
            focal_gamma=config.get("focal_gamma", 2.0),
            embedding_mixup=config.get("embedding_mixup", False),
            mixup_alpha=config.get("mixup_alpha", 0.2),
            label_smoothing=config.get("label_smoothing", 0.0),
            weight_decay=config.get("weight_decay", 0.01),
            n_heads=config.get("n_heads", 4),
            n_conv=config.get("n_conv", 2),
            n_attn=config.get("n_attn", 1),
        )

        # Reshape ACAV-style flat-feature mmap rows (B, F) into stacked windows
        # of length n. The original implementation looped in Python over each
        # window, which dominated per-step time at large batch sizes (~1024
        # iterations/step at ACAV=16384). A single reshape is ~100x faster and
        # keeps the GPU fed.
        def f(x, n=input_shape[0]):
            if x.shape[1] == n:
                return x
            x = np.vstack(x)
            n_full = (x.shape[0] // n) * n
            return x[:n_full].reshape(-1, n, x.shape[-1])

        # Create label transforms as needed for model (currently only supports binary classification models)
        data_transforms = {key: f for key in config["feature_data_files"].keys()}
        label_transforms = {}
        for key in ["positive"] + list(config["feature_data_files"].keys()) + ["adversarial_negative"]:
            if key == "positive":
                label_transforms[key] = lambda x: [1 for i in x]
            else:
                label_transforms[key] = lambda x: [0 for i in x]

        # Add generated positive and adversarial negative clips to the feature data files dictionary
        config["feature_data_files"]['positive'] = os.path.join(feature_save_dir, "positive_features_train.npy")
        config["feature_data_files"]['adversarial_negative'] = os.path.join(feature_save_dir, "negative_features_train.npy")

        # Make PyTorch data loaders for training and validation data
        batch_generator = mmap_batch_generator(
            config["feature_data_files"],
            n_per_class=config["batch_n_per_class"],
            data_transform_funcs=data_transforms,
            label_transform_funcs=label_transforms
        )

        class IterDataset(torch.utils.data.IterableDataset):
            def __init__(self, generator):
                self.generator = generator

            def __iter__(self):
                return self.generator

        # num_workers=0: forked workers each kept their own copy of
        # mmap_batch_generator with data_counter=0 and the DataLoader
        # round-robin produced N duplicate batches per step. mmap reads in
        # __next__ are sub-millisecond — workers add no throughput here.
        X_train = torch.utils.data.DataLoader(
            IterDataset(batch_generator),
            batch_size=None,
            num_workers=0,
        )

        # Validation DataLoaders: cap batch_size at 256. Previously these
        # used batch_size=len(labels), pushing the entire validation set
        # (up to ~hundreds of thousands of windows for X_val_fp) through
        # one model.forward call. The dnn head tolerated that; the
        # conv_attention head's MultiheadAttention crashes the CUDA SDPA
        # kernel with "invalid configuration argument" at large B (the
        # specific kernel-launch limit varies per GPU). 256 is well under
        # any backend limit and the val loop accumulates across batches.
        VAL_BATCH = 256

        X_val_fp = np.load(config["false_positive_validation_data_path"])
        # Each row = one 80 ms openwakeword embedding frame. Total audio
        # duration = n_frames * 0.08 s; convert to hours for FP/hr math.
        # Previously hardcoded to 11.3 (the upstream val set), which made
        # FP/hr meaningless for any other source.
        FRAME_SECONDS = 0.08
        val_set_hrs = X_val_fp.shape[0] * FRAME_SECONDS / 3600.0
        logging.info(
            "False-positive validation set: %d frames -> %.3f hours of audio",
            X_val_fp.shape[0], val_set_hrs,
        )
        X_val_fp = np.array([X_val_fp[i:i+input_shape[0]] for i in range(0, X_val_fp.shape[0]-input_shape[0], 1)])  # reshape to match model
        X_val_fp_labels = np.zeros(X_val_fp.shape[0]).astype(np.float32)
        X_val_fp = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)),
            batch_size=VAL_BATCH
        )

        X_val_pos = np.load(os.path.join(feature_save_dir, "positive_features_test.npy"))
        X_val_neg = np.load(os.path.join(feature_save_dir, "negative_features_test.npy"))
        labels = np.hstack((np.ones(X_val_pos.shape[0]), np.zeros(X_val_neg.shape[0]))).astype(np.float32)

        X_val = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(np.vstack((X_val_pos, X_val_neg))),
                torch.from_numpy(labels)
                ),
            batch_size=VAL_BATCH
        )

        # Run auto training
        best_model = oww.auto_train(
            X_train=X_train,
            X_val=X_val,
            false_positive_val_data=X_val_fp,
            steps=config["steps"],
            max_negative_weight=config["max_negative_weight"],
            target_fp_per_hour=config["target_false_positives_per_hour"],
            val_set_hrs=val_set_hrs,
        )

        # Export the trained model to onnx. Optional inference_temperature
        # (default 1.0 = identity) sharpens the sigmoid output post-hoc:
        # s' = sigmoid(T * logit(s)). T~11 maps a 0.55 peak to ~0.9.
        oww.export_model(
            model=best_model,
            model_name=config["model_name"],
            output_dir=config["output_dir"],
            temperature=float(config.get("inference_temperature", 1.0)),
        )
