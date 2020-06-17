"""
Store images from fashion MNIST in a Boltzmann network.
"""

import matplotlib.pyplot as plt
from tensorflow import keras

from boltzmann.boltzmann_machine import BoltzmannMachine
from preprocess.preprocess_image import *

print("Loading fashion MNIST...")
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocess
row, col = train_images[0].shape
data = train_images[5:9]
data = [normalize_binarize(i, 0) for i in data]
data = [i.flatten() for i in data]

# Train network
print("Training network...")
bm = BoltzmannMachine(n_visible_units=data[0].shape[0], n_hidden_units=500, n_epochs=1000, lr=0.01, batch_size=4)
bm.train(np.array(data))

# Get noisy data
noise_data = [noise_image(i, .3, min_val=0) for i in data]

# Run network
print("Running network on noisy data...")
data_hat = [bm.inference(i) for i in noise_data]

# Plot
print("Plotting results...")
total_data = data
total_data += noise_data
total_data += data_hat
plt.figure()
for i in range(1, len(total_data) + 1):
    plt.subplot(3, 4, i)
    plt.matshow(total_data[i - 1].reshape(row, col), fignum=False)
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.show()
