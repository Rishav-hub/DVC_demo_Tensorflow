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
    
    flatten_layer = tf.keras.layers.Flatten()

    predictions = tf.keras.layers.Dense(classes, activation='softmax')(flatten_layer(model.output))
    full_model = tf.keras.Model(inputs=model.input, outputs=predictions)


    full_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate= learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    logging.info("Custom model is compiled and ready to be trained")

    return full_model