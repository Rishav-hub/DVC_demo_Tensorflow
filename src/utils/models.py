import tensorflow as tf
import joblib
import logging
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.models import Model
from src.utils.all_utils import get_timestamp




def get_VGG_16_model(input_shape, model_path):
    """
    Returns a VGG16 model with the weights pre-trained on ImageNet.
    """
    model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    model.save(model_path)
    logging.info(f"VGG16 model loaded saved at {model_path}")
    return model

def prepare_model(model, classes,freeze_all, freeze_layers, learning_rate):
    """
    Compiles and returns a model.
    """
    if freeze_all:
        for layer in model.layers:
            layer.trainable = False
    else:
        for layer in model.layers[:freeze_layers]:
            layer.trainable = False
        for layer in model.layers[freeze_layers:]:
            layer.trainable = True
    model_seq = Sequential()

    model_seq.add(model)
    model_seq.add(tf.keras.layers.Flatten())
    model_seq.add(tf.keras.layers.Dense(classes, activation='softmax'))


    model_seq.compile(optimizer=tf.keras.optimizers.SGD(learning_rate= learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    logging.info("Custom model is compiled and ready to be trained")

    return model_seq

def load_full_model(model_path):
    """
    Loads a model from a file.
    """
    model = tf.keras.models.load_model(model_path)
    logging.info(f"Model loaded from {model_path}")
    return model

def get_unique_model_file_path(model_directory, model_name = "Model"):
    """
    Returns a unique file path for a model.
    """
    timestamp = get_timestamp(model_name)
    model_path = os.path.join(model_directory, f"{model_name}_{timestamp}.h5")
    return model_path