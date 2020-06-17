"""
Store images from fashion MNIST in a Hopfield network.
Training the network does not store the image itself, but updates the network's weights.
"""

from tensorflow import keras

from hopfield import Hopfield
from preprocess.preprocess_image import *
from utils.graphing import visualize

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

total_data = [i for i in data]
total_data += [i for i in noise_data]
total_data += data_hat

# Visualize inputs and outputs
visualize(np.array(total_data), n_plots=(3, 4), image_shape=(row, col))

# Visualize weights
visualize(hn.T, n_plots=(28, 28), image_shape=(row, col))
