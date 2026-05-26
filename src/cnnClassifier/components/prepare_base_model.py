import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

#model for different architectures
MODEL_CONFIGS = {
    "VGG16": {
        "fn":          tf.keras.applications.VGG16,
        "pooling":     "flatten",   # VGG16 spatial output suits Flatten
        "dropout":     0.5,
        "unfreeze":    0,           # 138M params — too large to fine-tune safely
        "lr_override": None,        # respects params.yaml LEARNING_RATE
        "optimizer":   "sgd" # VGG16 benefits from momentum and lower LR
    },
    "EfficientNetV2B3": {
        "fn":          tf.keras.applications.EfficientNetV2B3,
        "pooling":     "gap",       # GlobalAveragePooling2D — built for this
        "dropout":     0.4,
        "unfreeze":    20,          # unfreeze top 20 layers for CT domain adaptation
        "lr_override": 0.0001,      # overrides params.yaml — EfficientNet needs lower LR
        "optimizer":   "adam"
    },
    "DenseNet121": {
        "fn":          tf.keras.applications.DenseNet121,
        "pooling":     "gap",
        "dropout":     0.4,
        "unfreeze":    20,
        "lr_override": 0.0001,
        "optimizer":   "adam"
    },
    "ResNet50V2": {
        "fn":          tf.keras.applications.ResNet50V2,
        "pooling":     "gap",
        "dropout":     0.4,
        "unfreeze":    20,
        "lr_override": 0.0001,
        "optimizer":   "adam"
    },
}

DEFAULT_CONFIG = {
    "fn":          tf.keras.applications.VGG16,
    "pooling":     "gap",
    "dropout":     0.5,
    "unfreeze":    0,
    "lr_override": None,
    "optimizer":   "sgd"
}


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config     = config
        self.model_name = getattr(config, "params_model_name", "VGG16")
        self.model_cfg  = MODEL_CONFIGS.get(self.model_name, DEFAULT_CONFIG)

    def get_base_model(self):
        self.model = self.model_cfg["fn"](
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, learning_rate, model_cfg):
        # freeze all layers
        for layer in model.layers:
            layer.trainable = False

        #unfreeze top N layers for domain adaptation
        unfreeze = model_cfg["unfreeze"]
        if unfreeze > 0:
            for layer in model.layers[-unfreeze:]:
                layer.trainable = True

        #pooling Flatten for VGG16, GAP for all modern architectures
        if model_cfg["pooling"] == "flatten":
            x = tf.keras.layers.Flatten()(model.output)
        else:
            x = tf.keras.layers.GlobalAveragePooling2D()(model.output)

        #classification head
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(model_cfg["dropout"])(x)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(x)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        #lr_override takes priority over params.yaml value
        effective_lr = model_cfg["lr_override"] or learning_rate

        if model_cfg["optimizer"] == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=effective_lr)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=effective_lr)

        full_model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            learning_rate=self.config.params_learning_rate,
            model_cfg=self.model_cfg
        )
        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.full_model
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)