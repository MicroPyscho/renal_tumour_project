import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os
from cnnClassifier.utils.common import read_yaml
from pathlib import Path

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

PREPROCESSING_MAP = {
    "VGG16":            lambda x: x / 255.0,
    "EfficientNetV2B3": tf.keras.applications.efficientnet_v2.preprocess_input,
    "DenseNet121":      tf.keras.applications.densenet.preprocess_input,
    "ResNet50V2":       tf.keras.applications.resnet_v2.preprocess_input,
}

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        params = read_yaml(Path("params.yaml"))
        model_name = params.get("MODEL_NAME", "EfficientNetV2B3")
        model_path = os.path.join("artifacts", "training", f"{model_name}.keras")

        model = load_model(model_path)

        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # ✅ Apply correct preprocessing per model
        preprocess_fn = PREPROCESSING_MAP.get(model_name, lambda x: x / 255.0)
        test_image = preprocess_fn(test_image)

        predictions = model.predict(test_image)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions)) * 100

        return [{
            "image": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "all_scores": {
                CLASS_NAMES[i]: f"{float(predictions[0][i]) * 100:.2f}%"
                for i in range(len(CLASS_NAMES))
            }
        }]