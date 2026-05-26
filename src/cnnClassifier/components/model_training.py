import os
import urllib.request as request
from zipfile import ZipFile
from pathlib import Path
import tensorflow as tf
import time
from cnnClassifier.utils.common import build_datagenerator_kwargs
from cnnClassifier.entity.config_entity import TrainingConfig
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from cnnClassifier.components.prepare_base_model import MODEL_CONFIGS


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        
        learning_rate = self.config.all_params.get("LEARNING_RATE", 0.0001)
        model_name = self.config.all_params.get("MODEL_NAME", "VGG16")
        model_cfg = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["VGG16"])
        effective_lr = model_cfg["lr_override"] or learning_rate

        if model_cfg["optimizer"] == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=effective_lr)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=effective_lr)

        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        
#training section
    def train_valid_generator(self):
        
        model_name = self.config.all_params.get("MODEL_NAME", "VGG16")
        base_kwargs = build_datagenerator_kwargs(model_name)
        
        
        #data scaling and modification section
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        
#validation section
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator( 
            **base_kwargs
        )
        
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=Path(self.config.training_data)/"val",
            shuffle=False,
            class_mode="sparse",
            **dataflow_kwargs
            
        )
        
        if self.config.params_is_augmentation:
            model_name = self.config.all_params.get("MODEL_NAME", "VGG16")
            augmentation_kwargs = dict(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2
                )

            if model_name == "VGG16":
                augmentation_kwargs["brightness_range"] = [0.8, 1.2]

            combined_kwargs = {**augmentation_kwargs, **base_kwargs}
                
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                **combined_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator
            
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=Path(self.config.training_data)/"train",
            shuffle=True,
            class_mode="sparse",
            **dataflow_kwargs
        )

    def _compute_class_weights(self):
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        classes = np.unique(self.train_generator.classes)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=self.train_generator.classes
        )
        return dict(enumerate(weights))
        
    #for saving the model    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        
        
        
#training function
    def train(self):

        #generator reset to prevent overfit
        self.train_generator.reset()
        self.valid_generator.reset()

        class_weights = self._compute_class_weights()

        #safety net callbacks
        os.makedirs("checkpoints", exist_ok=True)

        callbacks = [
            ModelCheckpoint(
            filepath = f"checkpoints/best_{self.config.all_params.get('MODEL_NAME', 'model')}.keras",
            monitor = "val_loss",
            mode = "min",
            save_best_only = True,
            verbose = 1
        ),

        EarlyStopping(
            monitor = "val_loss",
            patience = 10,
            min_delta = 0.001,
            restore_best_weights = True,
            verbose = 1
        ),

        ReduceLROnPlateau(
            monitor = "val_loss",
            factor = 0.5,
            patience = 5,
            min_lr = 1e-7,
            verbose = 1
        )

        ]


        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            class_weight=class_weights,
            callbacks=callbacks
        )
        
#save the model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )