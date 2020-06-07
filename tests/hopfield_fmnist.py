"""
Store images from fashion MNIST in a Hopfield network.
Training the network does not store the image itself, but updates the network's weights.
"""

import sys
import os

from emergence.hopfield import *


import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

