import os
import pickle

from keras.datasets import mnist

from variational_autoencoder import VariationalAutoencoder
from ml_constants import DEFAULT_MODEL_PATH


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape(x_test.shape + (1,))

    return (x_train, y_train), (x_test, y_test)


def load_model(path=DEFAULT_MODEL_PATH):
    with open(os.path.join(path, "params.pkl"), "rb") as f:
        params = pickle.load(f)

    ae = VariationalAutoencoder(*params)
    ae.model.load_weights(os.path.join(path, "weights.h5"))

    return ae
