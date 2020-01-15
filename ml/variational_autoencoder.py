import os
import pickle

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Input,
    Lambda,
    LeakyReLU,
    Reshape,
)
from keras.models import Model
from keras.optimizers import Adam

from ml.constants import DEFAULT_MODEL_PATH


class VariationalAutoencoder:
    def __init__(
        self, input_dim, latent_dim, encoder_params=[], decoder_params=[],
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params

        self._build()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self.model = Model(self.encoder_input, self.decoder_model(self.encoder_output))

    def save(self, output_path=DEFAULT_MODEL_PATH):
        params = [
            self.input_dim,
            self.latent_dim,
            self.encoder_params,
            self.decoder_params,
        ]

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, "params.pkl"), "wb") as f:
            pickle.dump(params, f)

    def _build_encoder(self):
        self.encoder_input = Input(shape=self.input_dim, name="encoder_input")

        x = self.encoder_input

        for i, params in enumerate(self.encoder_params):
            x = Conv2D(**params, padding="same", name=f"encoder_conv_{i}")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)

        self.shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)

        # handle the variational bit
        self.mu = Dense(self.latent_dim, name="mu")(x)
        self.log_var = Dense(self.latent_dim, name="log_var")(x)

        self.encoder_mu_log_var = Model(self.encoder_input, (self.mu, self.log_var))

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
            return mu + K.exp(log_var / 2) * epsilon

        self.encoder_output = Lambda(sampling, name="encoder_output")(
            [self.mu, self.log_var]
        )
        self.encoder_model = Model(self.encoder_input, self.encoder_output)

    def _build_decoder(self):
        decoder_input = Input(shape=(self.latent_dim,), name="decoder_input")

        x = Dense(np.prod(self.shape_before_flattening))(decoder_input)
        x = Reshape(self.shape_before_flattening)(x)

        for i, params in enumerate(self.decoder_params):
            x = Conv2DTranspose(**params, padding="same", name=f"decoder_conv_t_{i}")(x)
            if i < len(self.decoder_params) - 1:
                x = BatchNormalization()(x)
                x = LeakyReLU()(x)

        decoder_output = Activation("sigmoid")(x)

        self.decoder_model = Model(decoder_input, decoder_output)

    def compile_model(self, lr, r_loss_factor):
        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
            return r_loss_factor * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss = -0.5 * K.sum(
                1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis=1
            )
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return r_loss + kl_loss

        optimizer = Adam(lr=lr)
        self.model.compile(
            optimizer=optimizer, loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss]
        )

    def train(self, train_data, batch_size, epochs, path=DEFAULT_MODEL_PATH):
        if not os.path.exists(path):
            os.makedirs(path)

        checkpoint = ModelCheckpoint(
            os.path.join(path, "weights.h5"),
            save_weights_only=True,
            verbose=1,
        )

        self.model.fit(
            train_data,
            train_data,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            callbacks=[checkpoint],
        )
