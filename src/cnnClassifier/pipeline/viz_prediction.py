import numpy as np
from pathlib import Path
from typing import Optional
 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
 
from cnnClassifier.pipeline.prediction import pixel_ct_gate   # reuse shared gate
 
 
# Preprocessing functions keyed by the string sent from the frontend
PREPROCESS_MAP = {
    "divide255":   lambda x: x / 255.0,
    "efficientnet": tf.keras.applications.efficientnet_v2.preprocess_input,
    "densenet":    tf.keras.applications.densenet.preprocess_input,
    "resnet_v2":   tf.keras.applications.resnet_v2.preprocess_input,
    "inception":   tf.keras.applications.inception_v3.preprocess_input,
}
 
# Cache loaded session models so repeated calls don't reload from disk
_MODEL_CACHE: dict[str, object] = {}
 
 
class VisPredictionPipeline:
    """
    Two-stage pipeline for visitor-uploaded models:
      Stage 1 — pixel CT gate  (same heuristic as AidRenal, always runs)
      Stage 2 — visitor's Keras model forward pass
    BiomedCLIP gate is intentionally skipped here because:
      - visitor models may be trained on non-CT medical images
      - gate should not block legitimate researcher demos
    """
 
    def __init__(
        self,
        filename:    str,
        model_path:  str,
        class_names: list[str],
        preprocess:  str = "divide255",
        img_size:    int = 224,
    ):
        self.filename    = filename
        self.model_path  = model_path
        self.class_names = class_names
        self.preprocess  = preprocess
        self.img_size    = img_size
 
    def _load_model(self):
        """Load from cache or disk."""
        if self.model_path not in _MODEL_CACHE:
            _MODEL_CACHE[self.model_path] = load_model(self.model_path)
        return _MODEL_CACHE[self.model_path]
 
    def predict(self) -> list[dict]:
        # ── Stage 1: pixel heuristic gate ──────────────────────────────
        from PIL import Image as PILImage
        pil_image = PILImage.open(self.filename).convert("RGB")
        img_array = np.array(pil_image)
 
        pixel_result = pixel_ct_gate(img_array)
        if not pixel_result["ok"]:
            return [{
                "gate_failed": True,
                "gate_stage":  "pixel_heuristic",
                "error":       pixel_result["msg"],
            }]
 
        # ── Stage 2: forward pass through visitor's model ───────────────
        model = self._load_model()
 
        test_image = keras_image.load_img(
            self.filename,
            target_size=(self.img_size, self.img_size)
        )
        test_image = keras_image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
 
        preprocess_fn = PREPROCESS_MAP.get(self.preprocess, lambda x: x / 255.0)
        test_image    = preprocess_fn(test_image)
 
        predictions    = model.predict(test_image)
        predicted_idx  = int(np.argmax(predictions, axis=1)[0])
 
        # Guard: model output may have more or fewer units than class_names
        n = min(len(self.class_names), predictions.shape[1])
        predicted_class = self.class_names[predicted_idx] if predicted_idx < n else f"class_{predicted_idx}"
        confidence      = float(np.max(predictions)) * 100
 
        return [{
            "image":      predicted_class,
            "confidence": f"{confidence:.2f}%",
            "all_scores": {
                self.class_names[i]: f"{float(predictions[0][i]) * 100:.2f}%"
                for i in range(n)
            },
            "model": "session_model",
            "gate_pixel": pixel_result["msg"],
        }]
 