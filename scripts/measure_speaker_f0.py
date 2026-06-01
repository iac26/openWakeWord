#!/usr/bin/env python3
"""Measure the median fundamental frequency (F0) of every speaker in a Piper
multi-speaker voice model and write a `<model>.f0.json` map next to it.

The map is consumed by openwakeword/train.py (`build_speaker_pool`) to
gender-weight the synthetic positive clips. The libritts-high pool is only
~12% female-range, which makes the trained wake word under-fire for women;
the F0 map lets training oversample higher-pitched speakers to a balanced
split.

The map is voice-model specific — rerun this whenever you change the Piper
voice. Output is `{speaker_id: median_f0_hz}` (speakers with too little voiced
audio to estimate are omitted).

Usage:
    uv run python scripts/measure_speaker_f0.py \
        --piper_dir ./training_workspace/piper-sample-generator
"""
import argparse
import json
import logging
import os
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--piper_dir", default="./training_workspace/piper-sample-generator",
                   help="Path to the piper-sample-generator checkout.")
    p.add_argument("--model", default=None,
                   help="Voice model .pt (default: <piper_dir>/models/en-us-libritts-high.pt).")
    p.add_argument("--phrase", default="hey there.",
                   help="Phrase to synthesize per speaker for F0 estimation.")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: <model>.f0.json).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--fmin", type=float, default=70.0)
    p.add_argument("--fmax", type=float, default=400.0)
    args = p.parse_args()

    model_path = args.model or os.path.join(args.piper_dir, "models", "en-us-libritts-high.pt")
    out_path = args.output or (model_path + ".f0.json")

    # Quiet the numba/librosa debug spam and let generate_samples load the
    # pickled Piper checkpoint under torch>=2.6 (weights_only defaults True).
    logging.disable(logging.WARNING)
    import torch
    _orig_load = torch.load
    torch.load = lambda *a, **k: (k.setdefault("weights_only", False), _orig_load(*a, **k))[1]

    sys.path.insert(0, os.path.abspath(args.piper_dir))
    import generate_samples as gs
    import torchaudio
    import librosa

    model = torch.load(model_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    cfg = json.load(open(model_path + ".json"))
    n_speakers = cfg["num_speakers"]
    phonemizer = gs.Phonemizer(cfg["espeak"]["voice"])
    phoneme_ids = gs.get_phonemes(phonemizer, cfg, args.phrase, False)
    resampler = torchaudio.transforms.Resample(cfg["audio"]["sample_rate"], 16000)

    print(f"Measuring F0 for {n_speakers} speakers in {model_path}", file=sys.stderr)
    results = {}
    for start in range(0, n_speakers, args.batch_size):
        batch = list(range(start, min(start + args.batch_size, n_speakers)))
        # Pure single-speaker timbre: speaker_1 == speaker_2 makes the SLERP
        # blend a no-op, so the measured F0 is the speaker's own.
        spk = torch.LongTensor(batch)
        with torch.no_grad():
            audio = gs.generate_audio(model, spk, spk, [phoneme_ids] * len(batch),
                                      0.5, 0.667, 0.8, 1.0, None)
        audio = resampler(audio.cpu()).numpy()
        for j, sid in enumerate(batch):
            y = audio[j].flatten().astype(np.float32)
            y = y / (np.abs(y).max() + 1e-9)
            f0, _, _ = librosa.pyin(y, fmin=args.fmin, fmax=args.fmax, sr=16000,
                                    frame_length=1024)
            voiced = f0[~np.isnan(f0)]
            results[sid] = round(float(np.median(voiced)), 1) if len(voiced) > 5 else None
        if start % (args.batch_size * 10) == 0:
            print(f"  {start}/{n_speakers}", file=sys.stderr)

    json.dump(results, open(out_path, "w"))
    measured = np.array([v for v in results.values() if v])
    print(f"Wrote {out_path} ({len(measured)}/{n_speakers} speakers measured)", file=sys.stderr)
    print(f"  median F0 {np.median(measured):.0f} Hz | "
          f">=165Hz {100 * np.mean(measured >= 165):.0f}% | "
          f">=200Hz {100 * np.mean(measured >= 200):.0f}%", file=sys.stderr)


if __name__ == "__main__":
    main()
