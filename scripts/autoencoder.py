"""
This script contains code used for constructing the autoencoder model to perform
the embedding for diagnosis and procedure codes. The embeddings of the diagnosis
and procedure codes are used as input for the transformer.py.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp
import seaborn as sns

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

def create_vae_none_sym(input_shape, neurons = 128, lr = 1e-6, latent_dim = 25):
    original_dim = input_shape

    inputs = keras.Input(shape=(original_dim,))

    x = layers.Dense(neurons,activation = "sigmoid")(inputs)
    x = layers.Dense(512,activation = "sigmoid")(x)
    x = layers.Dense(256,activation = "sigmoid")(x)

    z_mean = layers.Dense(latent_dim)(x)
    z_log_sigma = layers.Dense(latent_dim)(x)

    import tensorflow.keras.backend as K

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev = .1) # larger std
        return z_mean + K.exp(z_log_sigma) * epsilon # can add noise later

    z = layers.Lambda(sampling)([z_mean, z_log_sigma])

    # encoder
    encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

    # similarity loss
    similarity_loss = 0
    normalized_origin = tf.nn.l2_normalize(inputs, axis = 1)
    normalized_embd = tf.nn.l2_normalize(z_mean,axis = 1)

    # multiply row i with row j using transpose
    # element wise product
    prod_origin = tf.matmul(normalized_origin, normalized_origin,
                     adjoint_b = True # transpose second matrix
                     )

    prod_embd = tf.matmul(normalized_embd, normalized_embd,
                     adjoint_b = True # transpose second matrix
                     )

    origin_dist = 1 - prod_origin
    embd_dist = 1 - prod_embd
    similarity_loss = keras.losses.mean_squared_error(origin_dist, embd_dist)
    similarity_loss *= original_dim

    # decoder
    latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(256,activation = "sigmoid")(latent_inputs) # layer directly got samples from the learned distribution
    x = layers.Dense(512,activation = "sigmoid")(x)
    x = layers.Dense(neurons,activation = "sigmoid")(x)

    outputs = layers.Dense(original_dim)(latent_inputs)
    decoder = keras.Model(latent_inputs, outputs, name ='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = keras.Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = keras.losses.mean_squared_error(inputs, outputs)
    reconstruction_loss *= original_dim

    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss + similarity_loss)
    vae.add_loss(vae_loss)
    vae.add_metric(kl_loss,name = "kl loss")
    vae.add_metric(similarity_loss,name = "sim loss")
    vae.add_metric(reconstruction_loss,name="reconstruction loss")
    vae.compile(optimizer=Adam(learning_rate = lr))

    return vae, encoder, decoder

if __name__ == "__main__":

    from sklearn.preprocessing import normalize
    from tensorflow.keras.callbacks import ModelCheckpoint

    checkpoint = ModelCheckpoint(filepath="C:/Data/Lab/PatEmbedding/eMERGE/model/testrun/",
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=True,
                 save_weights_only=True,
                 mode='min')

    # define the model
    vae, encoder, decoder = create_vae_none_sym(input_shape = 4320,
                                            neurons = 1024,
                                            latent_dim = 50,
                                            lr = 1e-7)

    # load the matrix
    wdir = "C:/Data/Lab/PatEmbedding/eMERGE/"
    df_diags = pd.read_csv(wdir + "diags_week_mat.csv")
    df_diags.set_index("WEEK",inplace = True)
    df_procs = pd.read_csv(wdir + "procs_week_mat.csv",index_col=0)
    df_meds = pd.read_csv(wdir + "meds_week_mat.csv",index_col=0)

    # remove codes appeared less than 10 times
    def remove_counts(df, n=10):
        qual_columns = df.sum(axis=0)[df.sum(axis=0) >= 10].index.tolist()
        return df[qual_columns]

    df_diags = remove_counts(df_diags,n = 10)
    df_procs = remove_counts(df_procs,n = 10)
    df_meds = remove_counts(df_meds,n = 10)

    mat = pd.merge(df_diags.reset_index(),df_procs.reset_index(), on = "WEEK")
    mat = pd.merge(mat, df_meds.reset_index(), on = "WEEK", how = "outer").set_index("WEEK")
    varname = mat.columns.tolist()
    mat = mat.fillna(0.0).to_numpy()
    mat = normalize(mat.T, axis=0)

    print(mat.shape)

    vae.fit(mat, mat,
       batch_size = 64, epochs = 25,
       validation_split = .05,
       callbacks = [checkpoint])
