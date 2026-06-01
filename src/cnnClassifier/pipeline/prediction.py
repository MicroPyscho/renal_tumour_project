import os
import gc
import logging
import numpy as np
from pathlib import Path
from typing import Optional

import tensorflow as tf
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from cnnClassifier.utils.common import read_yaml

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = "/tmp/checkpoints"
_model_cache   = {}
_current_model = None

def download_model_if_missing(model_name: str) -> str:
    filename = f"best_{model_name}.keras"
    dest = os.path.join(CHECKPOINT_DIR, filename)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if not os.path.exists(dest):
        print(f"Downloading {filename}...")
        hf_hub_download(
            repo_id="MicroPyscho/AidRenal-models",
            filename=filename,
            repo_type="model",
            local_dir=CHECKPOINT_DIR,
        )
        print(f"✓ {filename} ready")
    return dest

def get_model(model_name: str):
    global _model_cache, _current_model
    if _current_model == model_name and model_name in _model_cache:
        return _model_cache[model_name]
    if _model_cache:
        print(f"Unloading {_current_model}...")
        _model_cache.clear()
        gc.collect()
        tf.keras.backend.clear_session()
    model_path = download_model_if_missing(model_name)
    print(f"Loading {model_name}...")
    model = load_model(model_path)
    _model_cache[model_name] = model
    _current_model = model_name
    print(f"✓ {model_name} active")
    return model

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

PREPROCESSING_MAP = {
    "VGG16":            lambda x: x / 255.0,
    "EfficientNetV2B3": tf.keras.applications.efficientnet_v2.preprocess_input,
    "DenseNet121":      tf.keras.applications.densenet.preprocess_input,
    "ResNet50V2":       tf.keras.applications.resnet_v2.preprocess_input,
}

CT_PROMPTS = [
    "an abdominal CT scan showing kidney",
    "a renal computed tomography scan",
    "a CT scan of the urinary tract",
]
NON_CT_PROMPTS = [
    "a colour photograph",
    "a diagram or illustration",
    "an MRI scan",
    "an X-ray radiograph",
    "a drawing or artwork",
    "a screenshot or document",
]
ALL_PROMPTS = CT_PROMPTS + NON_CT_PROMPTS
CT_SCORE_THRESHOLD = 0.35


class BiomedCLIPGate:
    _instance: Optional["BiomedCLIPGate"] = None
    _available: Optional[bool] = None
    
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = "cpu"

    @classmethod
    def get(cls) -> Optional["BiomedCLIPGate"]:
        if cls._available is False:
            return None
        if cls._instance is not None:
            return cls._instance
        gate = cls()
        try:
            import torch
            import open_clip
            gate.device = "cuda" if torch.cuda.is_available() else "cpu"
            gate.model, _, gate.preprocess = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            gate.tokenizer = open_clip.get_tokenizer(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            gate.model.to(gate.device).eval()
            cls._instance = gate
            cls._available = True
            logger.info("BiomedCLIP gate loaded on %s", gate.device)
        except Exception as exc:
            cls._available = False
            logger.warning("BiomedCLIP unavailable (%s). Using pixel heuristic only.", exc)
            return None
        return cls._instance

    def score(self, pil_image) -> tuple[float, str]:
        import torch
        img_tensor  = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        text_tokens = self.tokenizer(ALL_PROMPTS).to(self.device)
        with torch.no_grad():
            img_feats = self.model.encode_image(img_tensor)
            txt_feats = self.model.encode_text(text_tokens)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
            logits = (img_feats @ txt_feats.T).squeeze(0)
            probs  = logits.softmax(dim=0).cpu().numpy()
        ct_score   = float(probs[:len(CT_PROMPTS)].sum())
        top_non_ct = ALL_PROMPTS[len(CT_PROMPTS) + int(probs[len(CT_PROMPTS):].argmax())]
        reason = (f"BiomedCLIP CT score: {ct_score:.2f} (top non-CT: '{top_non_ct}')")
        return ct_score, reason


def pixel_ct_gate(img_array: np.ndarray) -> dict:
    from PIL import Image as PILImage
    pil = PILImage.fromarray(img_array.astype(np.uint8))
    pil = pil.resize((128, 128), PILImage.LANCZOS)
    arr = np.array(pil).astype(float)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    mx  = arr.max(axis=2)
    mn  = arr.min(axis=2)
    lum2 = (mx + mn) / 2
    denom = np.where(255 - np.abs(2*lum2 - 255) == 0, 1, 255 - np.abs(2*lum2 - 255))
    sat  = np.where(mx == 0, 0.0, (mx - mn) / denom)
    avg_sat = sat.mean()
    if avg_sat > 0.20:
        return {"ok": False, "warn": False,
                "msg": f"⛔ Colour photo detected (saturation {avg_sat*100:.0f}%) — renal CT scans are greyscale.",
                "stats": {"avg_sat": avg_sat}}
    lums   = 0.299*r + 0.587*g + 0.114*b
    n      = lums.size
    dark   = (lums <  20).sum() / n
    mid    = ((lums >= 20) & (lums <= 220)).sum() / n
    bright = (lums > 235).sum() / n
    bimodality = (dark + bright) / (mid + 0.001)
    if bimodality > 5.0:
        return {"ok": False, "warn": False,
                "msg": f"⛔ Image appears to be a diagram (bimodality {bimodality:.1f}).",
                "stats": {"bimodality": bimodality}}
    h_diff = np.abs(lums[:, :-1] - lums[:, 1:])
    v_diff = np.abs(lums[:-1, :] - lums[1:, :])
    hard_edge = ((h_diff > 150).sum() + (v_diff > 150).sum()) / (h_diff.size + v_diff.size)
    if hard_edge > 0.06:
        return {"ok": False, "warn": False,
                "msg": f"⛔ Hard pixel edges ({hard_edge*100:.1f}%) — looks like a diagram.",
                "stats": {"hard_edge": hard_edge}}
    if bright > 0.50:
        return {"ok": False, "warn": False,
                "msg": f"⛔ Predominantly white ({bright*100:.0f}% bright pixels).",
                "stats": {"bright": bright}}
    if mid < 0.15:
        return {"ok": False, "warn": False,
                "msg": f"⛔ Very little mid-grey content ({mid*100:.0f}%).",
                "stats": {"mid": mid}}
    if avg_sat > 0.10 or bimodality > 2.5:
        return {"ok": True, "warn": True,
                "msg": "⚠ Image accepted with caution — verify this is a renal CT slice.",
                "stats": {"avg_sat": avg_sat, "bimodality": bimodality}}
    return {"ok": True, "warn": False,
            "msg": f"✓ Pixel profile consistent with CT scan.",
            "stats": {"avg_sat": avg_sat, "mid": mid, "bimodality": bimodality}}


class PredictionPipeline:
    def __init__(self, filename: str):
        self.filename = filename

    def predict(self, requested_model: str = None) -> list[dict]:
        VALID_MODELS = {"DenseNet121", "ResNet50V2", "EfficientNetV2B3", "VGG16"}
        if requested_model and requested_model in VALID_MODELS:
            model_name = requested_model
        else:
            params = read_yaml(Path(os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "params.yaml")))
            model_name = params.get("MODEL_NAME", "EfficientNetV2B3")

        from PIL import Image as PILImage
        pil_image = PILImage.open(self.filename).convert("RGB")
        img_array = np.array(pil_image)

        pixel_result = pixel_ct_gate(img_array)
        if not pixel_result["ok"]:
            return [{"gate_failed": True, "gate_stage": "pixel_heuristic",
                     "error": pixel_result["msg"],
                     "stats": pixel_result.get("stats", {})}]

        clip_gate = BiomedCLIPGate.get()
        if clip_gate is not None:
            ct_score, clip_reason = clip_gate.score(pil_image)
            logger.info("BiomedCLIP: %s", clip_reason)
            if ct_score < CT_SCORE_THRESHOLD:
                return [{"gate_failed": True, "gate_stage": "biomedclip",
                         "error": f"⛔ BiomedCLIP CT confidence {ct_score:.0%} — upload a renal CT scan.",
                         "ct_score": ct_score, "reason": clip_reason}]

        try:
            model = get_model(model_name)
        except Exception as e:
            return [{"gate_failed": True, "gate_stage": "model_load",
                     "error": f"Failed to load {model_name}: {e}"}]

        test_image = keras_image.load_img(self.filename, target_size=(224, 224))
        test_image = keras_image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        preprocess_fn = PREPROCESSING_MAP.get(model_name, lambda x: x / 255.0)
        test_image    = preprocess_fn(test_image)

        predictions     = model.predict(test_image)
        predicted_idx   = int(np.argmax(predictions, axis=1)[0])
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = float(np.max(predictions)) * 100

        result = {
            "image":      predicted_class,
            "confidence": f"{confidence:.2f}%",
            "all_scores": {
                CLASS_NAMES[i]: f"{float(predictions[0][i]) * 100:.2f}%"
                for i in range(len(CLASS_NAMES))
            },
            "model": model_name,
        }
        result["gate_pixel"] = pixel_result["msg"]
        if clip_gate is not None:
            result["gate_clip_score"] = round(ct_score, 3)
        if pixel_result.get("warn"):
            result["gate_warning"] = pixel_result["msg"]
        return [result]