"""
Store images from fashion MNIST in a Boltzmann network.
"""

from tensorflow import keras

from emergence.boltzmann.boltzmann_machine import BoltzmannMachine
from emergence.utils.graphing import visualize
from emergence.preprocess.preprocess_image import *

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
bm = BoltzmannMachine(n_visible_units=data[0].shape[0], n_hidden_units=64, n_epochs=500, lr=0.01, batch_size=4)
bm.train(np.array(data))

# Get noisy data
noise_data = [noise_image(i, .3, min_val=0) for i in data]

# Run network
print("Running network on noisy data...")
data_hat = [bm.inference(i) for i in noise_data]

# Plot
total_data = [i for i in data]
total_data += [i for i in noise_data]
total_data += data_hat
visualize(np.array(total_data), n_plots=(3, 4), image_shape=(row, col))

visualize(np.array(bm.T).T, n_plots=(8, 8), image_shape=(row, col))
