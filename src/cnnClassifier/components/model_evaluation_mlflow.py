import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
import dagshub
import numpy as np
from urllib.parse import urlparse
from sklearn.metrics import classification_report, confusion_matrix

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import build_datagenerator_kwargs
from cnnClassifier.utils.common import read_yaml, create_directories, save_json


CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]   # ← match your folder names exactly


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        # Pull model name from params so every run self-labels correctly
        self.model_name = self.config.all_params.get("MODEL_NAME", "UnknownModel")

    #data generator
    def _valid_generator(self):
        model_name = self.config.all_params.get("MODEL_NAME", "VGG16")
        datagenerator_kwargs = build_datagenerator_kwargs(model_name)
        
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,          # must be False for confusion matrix to be meaningful
            class_mode="sparse",
            **dataflow_kwargs,
        )

    #model load
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    # evaluation
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self._compute_per_class_metrics()
        self.save_score()

    def _compute_per_class_metrics(self):
        """Predict full validation set and compute per-class recall/precision/F1."""
        self.valid_generator.reset()
        preds_raw = self.model.predict(self.valid_generator, verbose=1)
        y_pred = np.argmax(preds_raw, axis=1)
        y_true = self.valid_generator.classes

        report = classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        )
        self.cm = confusion_matrix(y_true, y_pred)

        # Flatten into loggable scalar metrics
        self.per_class_metrics = {}
        for cls in CLASS_NAMES:
            safe = cls.lower()
            self.per_class_metrics[f"precision_{safe}"] = report[cls]["precision"]
            self.per_class_metrics[f"recall_{safe}"]    = report[cls]["recall"]
            self.per_class_metrics[f"f1_{safe}"]        = report[cls]["f1-score"]

        self.per_class_metrics["macro_f1"]    = report["macro avg"]["f1-score"]
        self.per_class_metrics["weighted_f1"] = report["weighted avg"]["f1-score"]

    #save scores locally
    def save_score(self):
        scores = {
            "val_loss":     self.score[0],
            "val_accuracy": self.score[1],
            **self.per_class_metrics,
        }
        save_json(path=Path("scores.json"), data=scores)

    #mlflow logging
    def log_into_mlflow(self):
        dagshub.init(
            repo_owner="MicroPyscho",
            repo_name="renal_tumour_project",
            mlflow=True,
        )
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run(run_name=self.model_name):

            #params
            mlflow.log_params(self.config.all_params)

            #aggregate metrics
            mlflow.log_metrics({
                "val_loss":     self.score[0],
                "val_accuracy": self.score[1],
            })

            #per-class metrics
            mlflow.log_metrics(self.per_class_metrics)

            #confusion matrix as artifact
            cm_path = Path("confusion_matrix.txt")
            np.savetxt(cm_path, self.cm, fmt="%d")
            mlflow.log_artifact(str(cm_path))

            #model registry
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model,
                    name="model",                         # replaces deprecated artifact_path
                    registered_model_name=self.model_name # dynamic — reads from params.yaml
                )
            else:
                mlflow.keras.log_model(self.model, name="model")