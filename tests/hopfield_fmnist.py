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
data = train_images[5:9]
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
print("Plotting results...")
total_data = data
total_data += noise_data
total_data += data_hat
print("length total data: ", len(total_data))
plt.figure()
for i in range(1, len(total_data)+1):
    plt.subplot(3, 4, i)
    plt.matshow(total_data[i-1].reshape(row,col), fignum=False)
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
plt.show()