import os
from openwakeword.model import Model
from openwakeword.vad import VAD
from openwakeword.custom_verifier_model import train_custom_verifier

__all__ = ['Model', 'VAD', 'train_custom_verifier']

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "models")
_RELEASE_BASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"

FEATURE_MODELS = {
    "embedding": {
        "model_path": os.path.join(_MODELS_DIR, "embedding_model.onnx"),
        "download_url": f"{_RELEASE_BASE}/embedding_model.onnx",
    },
    "melspectrogram": {
        "model_path": os.path.join(_MODELS_DIR, "melspectrogram.onnx"),
        "download_url": f"{_RELEASE_BASE}/melspectrogram.onnx",
    },
}

VAD_MODELS = {
    "silero_vad": {
        "model_path": os.path.join(_MODELS_DIR, "silero_vad.onnx"),
        "download_url": f"{_RELEASE_BASE}/silero_vad.onnx",
    }
}

MODELS = {
    name: {
        "model_path": os.path.join(_MODELS_DIR, f"{name}_v0.1.onnx"),
        "download_url": f"{_RELEASE_BASE}/{name}_v0.1.onnx",
    }
    for name in ("alexa", "hey_mycroft", "hey_jarvis", "hey_rhasspy", "timer", "weather")
}

model_class_mappings = {
    "timer": {
        "1": "1_minute_timer",
        "2": "5_minute_timer",
        "3": "10_minute_timer",
        "4": "20_minute_timer",
        "5": "30_minute_timer",
        "6": "1_hour_timer"
    }
}


def get_pretrained_model_paths(inference_framework="onnx"):
    if inference_framework != "onnx":
        raise ValueError(f"Unsupported inference_framework={inference_framework!r}; only 'onnx' is supported")
    return [MODELS[i]["model_path"] for i in MODELS.keys()]
