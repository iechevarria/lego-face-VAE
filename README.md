# Lego face VAE

Variational autoencoder (VAE) for Lego minifig faces

## Overview

This repo contains a training set of images of Lego minifig faces and code to
train a VAE on them. 

Much of the code defining the VAE model is derived from 
[David Foster](https://github.com/davidADSP)'s excellent book
[*Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play*](https://www.amazon.com/Generative-Deep-Learning-Teaching-Machines/dp/1492041947) and from the book's
accompanying [repository](https://github.com/davidADSP/GDL_code).

### Sample Output

The following is a plot of random images that were reconstructed by the VAE. The
top image is the input and the bottom image is the VAE's reconstruction: 

<img src="./docs/img/reconstruction.png">

The following are plots of the intermediate vectors between face encodings (face
morph visualization):

<img src="./docs/img/face-morph-1.png">
<img src="./docs/img/face-morph-2.png">
<img src="./docs/img/face-morph-3.png">

## Quickstart on Google Colab

[Google Colab](https://colab.research.google.com/) is a free environment to run
Jupyter Notebooks with the option of using GPU/TPU instances.

To run the notebook in Colab, first go to
[https://colab.research.google.com/github/iechevarria/lego-face-VAE/blob/master/VAE_colab.ipynb](https://colab.research.google.com/github/iechevarria/lego-face-VAE/blob/master/VAE_colab.ipynb).

Next, run the following commands that appear in the section __"Set up Colab
environment"__:

```
!git clone https://github.com/iechevarria/lego-face-VAE
cd lego-face-VAE
!unzip dataset.zip
```

It should now be possible to run all the sections in the notebook. If you want
to experiment with the pretrained model included in the repo, skip to the
notebook section titled __"Do the fun stuff with the VAE"__. If you want to
train your own VAE, run the cells in the section titled
__"Train VAE on Lego faces"__. 

## Contents

The following table lays out directories/files and their purposes:

| directory/file                  | description                                            |
|---------------------------------|--------------------------------------------------------|
| dataset_scripts/                | Scripts to pull and process dataset images             |
| ml/utils.py                     | Utilities for loading data and for making plots        |
| ml/variational_autoencoder.py   | Defines the VAE model                                  |
| trained_model/                  | Pretrained model params and weights                    |
| VAE_colab.ipynb                 | Notebook to train and evaluate models                  |
| dataset.zip                     | Zipped directory of training images                    |

## Training set details

The training data (approximately 3800 128x128 JPEG images of Lego minifig faces)
is contained in `dataset.zip`. These images were pulled from a 
[Quartz aticle](https://qz.com/1405657/how-are-lego-emotions-changing-help-us-find-out/)
and [Bricklink](https://www.bricklink.com/). The photographs from the Quartz
article were taken by [Christoph Bartneck, Ph.D.](http://www.bartneck.de/)

The scripts used to pull and process the data are contained in the
`dataset_scripts` directory. I manually removed images that were low quality
or that did not contain a clear face.
