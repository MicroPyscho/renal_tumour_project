import os
import urllib.request as request
from zipfile import ZipFile
from pathlib import Path
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        
#training section
    def train_valid_generator(self):
        
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )
        
        #data scaling and modification section
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        
#validation section
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator( 
            **datagenerator_kwargs
        )
        
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            class_mode="sparse",
            **dataflow_kwargs
            
        )
        
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                brightness_range=[0.8, 1.2],
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator
            
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            class_mode="sparse",
            **dataflow_kwargs
        )
        
    #for saving the model    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        
        
        
#training function
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            workers=1,
            use_multiprocessing=False    
        )
        
#save the model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )