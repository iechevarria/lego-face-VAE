import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from PIL import Image

from ml.constants import DEFAULT_MODEL_PATH
from ml.variational_autoencoder import VariationalAutoencoder


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape(x_test.shape + (1,))

    return (x_train, y_train), (x_test, y_test)


def load_lego_faces(path="dataset", size=32):
    faces = {}
    for fname in os.listdir(path):
        with Image.open(os.path.join(path, fname)) as raw:
            img = raw.resize((size, size), Image.ANTIALIAS)
            img = np.divide(np.array(img), 255)
            faces[fname] = img

    return faces


def plot_reconstructed_images(data, encoder, decoder, n_to_show=10):
    example_idx = np.random.choice(range(len(data)), n_to_show)
    example_images = data[example_idx]

    latent_coords = encoder.predict(example_images)
    reconst_images = decoder.predict(latent_coords)

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(n_to_show):
        img = example_images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i + 1)
        sub.axis("off")
        sub.imshow(img)

    for i in range(n_to_show):
        img = reconst_images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i + n_to_show + 1)
        sub.axis("off")
        sub.imshow(img)


def morph_images(image_1, image_2, encoder, decoder, n_steps=10):
    vec1 = encoder.predict(np.array([image_1]))
    vec2 = encoder.predict(np.array([image_2]))

    morph_vecs = [
        np.add(
            np.multiply(vec1, (n_steps - 1 - i) / (n_steps - 1)),
            np.multiply(vec2, i / (n_steps - 1)),
        )
        for i in range(n_steps)
    ]

    morph_imgs = [decoder.predict(vec) for vec in morph_vecs]

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, morph_img in enumerate(morph_imgs):
        img = morph_img.squeeze()
        sub = fig.add_subplot(1, n_steps, i + 1)
        sub.axis("off")
        sub.imshow(img)


def load_model(path=DEFAULT_MODEL_PATH):
    with open(os.path.join(path, "params.pkl"), "rb") as f:
        params = pickle.load(f)

    ae = VariationalAutoencoder(*params)
    ae.model.load_weights(os.path.join(path, "weights.h5"))

    return ae
