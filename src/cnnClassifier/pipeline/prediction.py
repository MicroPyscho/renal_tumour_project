import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from cnnClassifier.utils.common import read_yaml
from pathlib import Path

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]  # must match folder order

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model name from params.yaml to always use the correct model
        params = read_yaml(Path("params.yaml"))
        model_name = params.get("MODEL_NAME", "EfficientNetV2B3")
        model_path = os.path.join("artifacts", "training", f"{model_name}.keras")

        model = load_model(model_path)

        # Preprocess image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # normalise to [0, 1]

        # Predict
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