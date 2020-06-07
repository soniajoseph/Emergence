"""
Store images from fashion MNIST in a Hopfield network.
Training the network does not store the image itself, but updates the network's weights.
"""

import sys
import os

from emergence.hopfield import Hopfield
from emergence.preprocess.preprocess_image import * 

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

print("Loading fashion MNIST...")
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocess
row, col = train_images[0].shape
data = train_images[:10]
data = [normalize_binarize(i) for i in data]
data = [i.flatten() for i in data]

# Train network
print("Training network...")
hn = Hopfield()
hn.train(data)

# Get noisy data
noise_data = [noise_image(i, .3) for i in data]

# Run network
print("Running network on noisy data...")
data_hat = [hn.run(i, 1) for i in noise_data]

# Plot
for i in range(1, 31):
    plt.subplot(3, 10, i)
plt.show()