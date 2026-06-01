"""
AidRenal · Prediction Pipeline
===============================
Two-stage prediction:
  Stage 1 — CT Gate:  rejects non-CT images before they reach the classifier.
             Tier A (if available): BiomedCLIP vision-language model (semantic).
             Tier B (always):       pixel heuristic — histogram + bimodality + edge analysis.
  Stage 2 — Renal Classifier: DenseNet121 / ResNet50V2 / EfficientNetV2B3 / VGG16.

Install BiomedCLIP once on your machine (requires internet, ~900 MB):
    pip install open-clip-torch
    python -c "
        import open_clip
        open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
    "
After that it is cached and works offline.
"""

import os
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

def download_models_if_missing():
    models = [
        "best_DenseNet121.keras",
        "best_ResNet50V2.keras", 
        "best_EfficientNetV2B3.keras",
    ]
    os.makedirs("/tmp/checkpoints", exist_ok=True)
    for filename in models:
        dest = f"checkpoints/{filename}"
        if not os.path.exists(dest):
            print(f"Downloading {filename}...")
            hf_hub_download(
                repo_id="MicroPyscho/AidRenal-models",
                filename=filename,
                repo_type="model",
                local_dir="/tmp/checkpoints"
            )
            print(f"✓ {filename} ready")

download_models_if_missing()

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

PREPROCESSING_MAP = {
    "VGG16":            lambda x: x / 255.0,
    "EfficientNetV2B3": tf.keras.applications.efficientnet_v2.preprocess_input,
    "DenseNet121":      tf.keras.applications.densenet.preprocess_input,
    "ResNet50V2":       tf.keras.applications.resnet_v2.preprocess_input,
}

# BiomedCLIP prompts — order matters: CT prompts first
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
CT_SCORE_THRESHOLD = 0.35   # sum of CT-label softmax scores needed to pass gate


# ── Stage 1A: BiomedCLIP gate ──────────────────────────────────────────────────
class BiomedCLIPGate:
    """
    Semantic CT gate using Microsoft BiomedCLIP.
    Loaded once and cached; gracefully unavailable if model not downloaded.
    """
    _instance: Optional["BiomedCLIPGate"] = None
    _available: Optional[bool] = None

    def __init__(self):
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = "cpu"

    @classmethod
    def get(cls) -> Optional["BiomedCLIPGate"]:
        """Singleton — load once, reuse across requests."""
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
            logger.warning(
                "BiomedCLIP unavailable (%s). "
                "Using pixel heuristic gate only. "
                "Run  pip install open-clip-torch  then pre-download the model "
                "to enable semantic CT validation.",
                exc
            )
            return None

        return cls._instance

    def score(self, pil_image) -> tuple[float, str]:
        """
        Returns (ct_score, reason).
        ct_score in [0, 1] — probability mass on CT-label prompts.
        """
        import torch
        img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        text_tokens = self.tokenizer(ALL_PROMPTS).to(self.device)

        with torch.no_grad():
            img_feats  = self.model.encode_image(img_tensor)
            txt_feats  = self.model.encode_text(text_tokens)
            img_feats  /= img_feats.norm(dim=-1, keepdim=True)
            txt_feats  /= txt_feats.norm(dim=-1, keepdim=True)
            logits     = (img_feats @ txt_feats.T).squeeze(0)
            probs      = logits.softmax(dim=0).cpu().numpy()

        ct_score = float(probs[:len(CT_PROMPTS)].sum())
        top_non_ct = ALL_PROMPTS[len(CT_PROMPTS) + int(probs[len(CT_PROMPTS):].argmax())]
        reason = (
            f"BiomedCLIP CT score: {ct_score:.2f} "
            f"(top non-CT match: '{top_non_ct}')"
        )
        return ct_score, reason


# ── Stage 1B: Pixel heuristic gate ────────────────────────────────────────────
def pixel_ct_gate(img_array: np.ndarray) -> dict:
    """
    Fast pixel-statistics check. Runs always — first line of defence.
    Returns dict with keys: ok (bool), warn (bool), msg (str), stats (dict).

    Checks (in order):
      1. Colour saturation    — rejects colour photographs
      2. Bimodality index     — rejects SVGs, logos, diagrams (near-binary pixel values)
      3. Hard-edge ratio      — rejects vector-art renders
      4. Bright-pixel ratio   — rejects white-background illustrations
      5. Mid-grey content     — requires real tissue gradient content
    """
    # Resize to 128×128 for speed (PIL-free, pure numpy)
    from PIL import Image as PILImage
    pil = PILImage.fromarray(img_array.astype(np.uint8))
    pil = pil.resize((128, 128), PILImage.LANCZOS)
    arr = np.array(pil).astype(float)

    # Handle greyscale-stored-as-RGB
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    # 1 — Saturation
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    lum2 = (mx + mn) / 2
    denom = np.where(255 - np.abs(2*lum2 - 255) == 0, 1,
                     255 - np.abs(2*lum2 - 255))
    sat = np.where(mx == 0, 0.0, (mx - mn) / denom)
    avg_sat = sat.mean()
    if avg_sat > 0.20:
        return {
            "ok": False, "warn": False,
            "msg": (
                f"⛔ Colour photo detected (saturation {avg_sat*100:.0f}%) — "
                "renal CT scans are greyscale."
            ),
            "stats": {"avg_sat": avg_sat},
        }

    # Luminance
    lums = 0.299*r + 0.587*g + 0.114*b
    n = lums.size
    dark   = (lums <  20).sum() / n   # near-black  (air / background)
    mid    = ((lums >= 20) & (lums <= 220)).sum() / n   # tissue grey range
    bright = (lums > 235).sum() / n   # near-white

    # 2 — Bimodality (SVGs: dark+bright dominant, almost no mid)
    bimodality = (dark + bright) / (mid + 0.001)
    if bimodality > 5.0:
        return {
            "ok": False, "warn": False,
            "msg": (
                f"⛔ Image appears to be a diagram or illustration — "
                f"pixel values are almost entirely pure black or white "
                f"(bimodality index {bimodality:.1f}). "
                "CT scans have rich mid-grey tissue gradients."
            ),
            "stats": {"bimodality": bimodality, "dark": dark,
                      "mid": mid, "bright": bright},
        }

    # 3 — Hard edges (vector art = abrupt luminance jumps)
    h_diff = np.abs(lums[:, :-1] - lums[:, 1:])   # horizontal neighbours
    v_diff = np.abs(lums[:-1, :] - lums[1:, :])   # vertical neighbours
    hard_edge = ((h_diff > 150).sum() + (v_diff > 150).sum()) / (h_diff.size + v_diff.size)
    if hard_edge > 0.06:
        return {
            "ok": False, "warn": False,
            "msg": (
                f"⛔ Hard pixel edges detected ({hard_edge*100:.1f}% of adjacent pairs) — "
                "this looks like a diagram or vector illustration. "
                "CT scan images have smooth gradient transitions between tissue types."
            ),
            "stats": {"hard_edge": hard_edge},
        }

    # 4 — White-background rejection
    if bright > 0.50:
        return {
            "ok": False, "warn": False,
            "msg": (
                f"⛔ Image is predominantly white ({bright*100:.0f}% bright pixels) — "
                "CT scans have significant dark regions from surrounding air."
            ),
            "stats": {"bright": bright},
        }

    # 5 — Mid-grey content (real CT must have tissue range)
    if mid < 0.15:
        return {
            "ok": False, "warn": False,
            "msg": (
                f"⛔ Very little mid-grey content ({mid*100:.0f}%) — "
                "CT scans have tissue, fat and organ structures producing "
                "a wide range of grey values."
            ),
            "stats": {"mid": mid},
        }

    # Marginal — pass with warning
    if avg_sat > 0.10 or bimodality > 2.5:
        return {
            "ok": True, "warn": True,
            "msg": (
                "⚠ Image accepted with caution — verify this is a "
                "renal/abdominal CT slice before classifying."
            ),
            "stats": {"avg_sat": avg_sat, "bimodality": bimodality},
        }

    return {
        "ok": True, "warn": False,
        "msg": (
            f"✓ Pixel profile consistent with CT scan "
            f"(sat {avg_sat*100:.1f}%, mid-grey {mid*100:.0f}%, "
            f"bimodality {bimodality:.2f})."
        ),
        "stats": {"avg_sat": avg_sat, "mid": mid, "bimodality": bimodality},
    }


# ── Stage 2: Renal Classifier ─────────────────────────────────────────────────
class PredictionPipeline:
    """
    Full two-stage pipeline:
      Stage 1 — CT gate (pixel heuristic + optional BiomedCLIP)
      Stage 2 — 4-class renal classifier
    """

    def __init__(self, filename: str):
        self.filename = filename

    def predict(self, requested_model: str = None) -> list[dict]:
        # ── Resolve model name ────────────────────────────────────────────────
        # Priority: 1) caller passes a model name (from frontend selector)
        #           2) params.yaml MODEL_NAME
        #           3) fallback to EfficientNetV2B3
        VALID_MODELS = {"DenseNet121", "ResNet50V2", "EfficientNetV2B3", "VGG16"}

        if requested_model and requested_model in VALID_MODELS:
            model_name = requested_model
        else:
            params     = read_yaml(Path("params.yaml"))
            model_name = params.get("MODEL_NAME", "EfficientNetV2B3")

        # Check the file actually exists — fall back to any available model
        # Name variants to try for each architecture
        NAME_VARIANTS = {
            "DenseNet121":     ["DenseNet121", "best_DenseNet121"],
            "ResNet50V2":      ["ResNet50V2",  "best_ResNet50v2", "best_ResNet50V2"],
            "EfficientNetV2B3":["EfficientNetV2B3", "best_EfficientNetV2B3"],
            "VGG16":           ["VGG16", "best_VGG16", "model"],
        }
        SEARCH_DIRS = [
            os.path.join("artifacts", "training"),
            "/tmp/checkpoints",
            os.path.join("/tmp/checkpoints", "VGG16"),
        ]

        model_path = None
        for variant in NAME_VARIANTS.get(model_name, [model_name]):
            for d in SEARCH_DIRS:
                candidate = os.path.join(d, f"{variant}.keras")
                if os.path.exists(candidate):
                    model_path = candidate
                    break
            if model_path:
                break

        if not model_path:
            # Last resort: any .keras file in either directory
            import glob
            all_models = (
                glob.glob(os.path.join("artifacts", "training", "*.keras")) +
                glob.glob(os.path.join("/tmp/checkpoints", "*.keras"))
            )
            if not all_models:
                return [{"gate_failed": True, "gate_stage": "model_load",
                         "error": "No trained model found in artifacts/training/ or checkpoints/. Run dvc repro first."}]
            model_path = all_models[0]
            model_name = os.path.basename(model_path).replace(".keras", "").replace("best_", "")
            import logging
            logging.getLogger(__name__).warning(
                "Requested model %s not found, falling back to %s at %s",
                requested_model or "params.yaml", model_name, model_path
            )

        # ── Load raw image for gate ───────────────────────────────────────────
        from PIL import Image as PILImage
        pil_image = PILImage.open(self.filename).convert("RGB")
        img_array = np.array(pil_image)

        # ── Stage 1B: Pixel heuristic (always runs) ──────────────────────────
        pixel_result = pixel_ct_gate(img_array)
        if not pixel_result["ok"]:
            return [{
                "gate_failed": True,
                "gate_stage":  "pixel_heuristic",
                "error":       pixel_result["msg"],
                "stats":       pixel_result.get("stats", {}),
            }]

        # ── Stage 1A: BiomedCLIP (runs if model is cached) ───────────────────
        clip_gate = BiomedCLIPGate.get()
        if clip_gate is not None:
            ct_score, clip_reason = clip_gate.score(pil_image)
            logger.info("BiomedCLIP: %s", clip_reason)
            if ct_score < CT_SCORE_THRESHOLD:
                return [{
                    "gate_failed": True,
                    "gate_stage":  "biomedclip",
                    "error": (
                        f"⛔ BiomedCLIP does not recognise this as a CT scan "
                        f"(CT confidence {ct_score:.0%}). "
                        "Please upload a renal abdominal CT scan."
                    ),
                    "ct_score": ct_score,
                    "reason":   clip_reason,
                }]

        # ── Stage 2: Renal classification ────────────────────────────────────
        model = load_model(model_path)

        test_image = keras_image.load_img(self.filename, target_size=(224, 224))
        test_image = keras_image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        preprocess_fn = PREPROCESSING_MAP.get(model_name, lambda x: x / 255.0)
        test_image    = preprocess_fn(test_image)

        predictions    = model.predict(test_image)
        predicted_idx  = int(np.argmax(predictions, axis=1)[0])
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

        # Attach gate metadata
        result["gate_pixel"] = pixel_result["msg"]
        if clip_gate is not None:
            result["gate_clip_score"] = round(ct_score, 3)

        # Pass-through warning from pixel gate
        if pixel_result.get("warn"):
            result["gate_warning"] = pixel_result["msg"]

        return [result]