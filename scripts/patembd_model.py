"""
The patient embedding model architecture code.
Some code (functions) are directly borrowed from the tensorflow website
https://www.tensorflow.org/text/tutorials/transformer
for reference purposes.
"""

# import
import os
import numpy as np
import pandas as pd
import scipy.stats as sp

import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

# functions


# model architecture
def patient_embedding_model(embd_size):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Add, Subtract, Concatenate, Dropout
    import tensorflow.keras.backend as K
    pre_embd_layer = keras.layers.Dense(1024,activation = "relu",name = "pre_embd")
    #pre_embd_layer2 = keras.layers.Dense(128,activation = "relu")
    embd_layer = keras.layers.Dense(embd_size,activation = "sigmoid",name = "embd")
    mean_pool = keras.layers.GlobalAveragePooling1D(data_format="channels_last",
                                                   #keepdims=False,
                                                  )
    # maybe mean pool isn't the best way
    model1 = keras.Sequential([
        keras.Input(shape=(250,50)),
        mean_pool,
        pre_embd_layer,
        #pre_embd_layer2,
        embd_layer
    ])

    model2 = keras.Sequential([
        keras.Input(shape=(250,50)),
        mean_pool,
        pre_embd_layer,
        #pre_embd_layer2,
        embd_layer
    ])

    similarity_measure = keras.metrics.mean_squared_error(model1.output,
                                                        model2.output)
    similarity_measure = K.log(1e-8+similarity_measure)

    output_layer_concat = Concatenate()([model1.output, model2.output])
    output_layer_concat = keras.layers.Dense(512,activation="relu")(output_layer_concat)
    output_layer_concat = keras.layers.Dense(128,activation="relu")(output_layer_concat)
    output_layer_concat = Dropout(rate=.2)(output_layer_concat)
    # patient_pred = keras.layers.Dense(1)(output_layer_concat)
    next_event_pred = keras.layers.Dense(1)(output_layer_concat)


    combine = Model([model1.input, model2.input], [next_event_pred, similarity_measure])
    return combine


def loss_function(real, pred):

    loss_ = loss_object(real, pred)

    return tf.reduce_sum(loss_)

loss_object = tf.keras.losses.BinaryCrossentropy(
    from_logits=True,reduction='none')

def loss_function_logits(real, pred):

    loss_ = loss_object_logit(real, pred)

    return tf.reduce_sum(loss_)

loss_object_logit = tf.keras.losses.BinaryCrossentropy(
    from_logits=True,reduction='none')
