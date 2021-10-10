import tensorflow as tf
import joblib
import logging
import os
from tensorflow.keras.applications.vgg16 import VGG16





def get_VGG_16_model(input_shape, model_path):
    """
    Returns a VGG16 model with the weights pre-trained on ImageNet.
    """
    model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    model.load_weights(model_path)
    return model