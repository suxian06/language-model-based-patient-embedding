

if __name__ == "__main__":

    import autoencoder
    import numpy as np
    from sklearn.preprocessing import normalize
    from tensorflow.keras.callbacks import ModelCheckpoint

    checkpoint = ModelCheckpoint(filepath="C:/Data/Lab/PatEmbedding/eMERGE/model/testrun/",
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=True,
                 save_weights_only=True,
                 mode='min')

    # define the model
    vae, encoder, decoder = autoencoder.create_vae_none_sym(input_shape = 4320,
                                                neurons = 1024,
                                                latent_dim = 50,
                                                lr = 1e-7)

    vae.load_weights("C:/Data/Lab/PatEmbedding/eMERGE/model/testrun/")
    mat = np.load("./example_data/mat.npy")

    encoded_feature = encoder.predict(mat)[0]
    np.save("./example_data/mat_embedded.npy",encoded_feature)
