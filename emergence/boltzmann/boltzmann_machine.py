import numpy as np
import tensorflow as tf

from emergence.hopfield.hopfield import Hopfield


class BoltzmannMachine(Hopfield):
    def __init__(self, n_visible_units, n_hidden_units, k=5, lr=1, n_epochs=10, batch_size=1):
        """
        Boltzmann machine network
        Args:
            n_visible_units (int): Number of visible units
            n_hidden_units (int): Number of hidden units
            k (int): k-folds for Gibbs sampling
            lr (float): learning rate
            n_epochs (int): number of epochs
            batch_size (int): batch size
        """
        super(BoltzmannMachine, self).__init__()
        self.n_visible_units = n_visible_units
        self.n_hidden_units = n_hidden_units
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.k = k
        self.T = tf.random.uniform((self.n_visible_units, self.n_hidden_units), minval=-0.1, maxval=0.1,
                                   dtype='float32')
        self.v_bias = tf.zeros(n_visible_units, dtype='float32')
        self.h_bias = tf.zeros(n_hidden_units, dtype='float32')

    def sample_hidden(self, v):
        """
        Visible units binomial sampling
        Args:
            v (tensor): input data

        Returns:
            tensor, tensor
        """
        probability_h_given_v = tf.sigmoid(v @ self.T + self.h_bias)
        return probability_h_given_v, tf.keras.backend.random_binomial(probability_h_given_v.shape,
                                                                       probability_h_given_v)

    def sample_visible(self, h):
        """
        Hidden units binomial sampling
        Args:
            h (tensor): hidden units data

        Returns:
            tensor, tensor
        """
        probability_v_given_h = tf.sigmoid(h @ tf.transpose(self.T) + self.v_bias)
        return probability_v_given_h, tf.keras.backend.random_binomial(probability_v_given_h.shape,
                                                                       probability_v_given_h)

    def update(self, v0, vk, p_h0, p_hk):
        """
        Update weights and biases
        Args:
            v0 (tensor): Original input data
            vk (tensor): Sampled input data
            p_h0 (tensor): Probability of h = 1 given v0
            p_hk (tensor): Probability of h = 1 given vk

        """
        self.T += self.lr * (tf.transpose(v0) @ p_h0 - tf.transpose(vk) @ p_hk) / self.batch_size
        self.v_bias += self.lr * (tf.reduce_mean(v0 - vk, axis=0))
        self.h_bias += self.lr * (tf.reduce_mean(p_h0 - p_hk, axis=0))

    def contrastive_divergence(self, v):
        """
        Contrastive divergence
        Args:
            v (tensor): input data

        Returns:
            tensor, tensor: original input data and sampled data
        """
        vk = v
        ph_v, _ = self.sample_hidden(v)

        # Gibbs sampling
        for k in range(self.k):
            _, hk = self.sample_hidden(vk)
            _, vk = self.sample_visible(hk)

        ph_k, _ = self.sample_hidden(vk)
        self.update(v, vk, ph_v, ph_k)
        return v, vk

    def energy(self, v):
        """
        Network energy
        Args:
            v (tensor): input data

        Returns:
            float: energy
        """
        _, hs = self.sample_hidden(v)
        e = - v @ tf.expand_dims(self.v_bias, 1) - hs @ tf.expand_dims(self.h_bias, 1) - v @ self.T @ tf.transpose(hs)
        return tf.reduce_sum(e)

    def train(self, data):
        """
        Train network
        Args:
            data (nd.array): input data
        """
        for i in range(self.n_epochs):
            loss = 0
            for j in range(0, len(data), self.batch_size):
                v_j = data[j:j + self.batch_size, :]
                v_j = tf.convert_to_tensor(v_j, dtype='float32')
                v0, vk = self.contrastive_divergence(v_j)
                loss += tf.reduce_mean(tf.abs(v0 - vk))
            print(f'epoch {i} - loss: {loss}')

    def inference(self, v):
        """
        Inference
        Args:
            v (tensor): input data

        Returns:
            nd.array: sampled output
        """
        v = tf.convert_to_tensor(v, dtype='float32')
        hp, _ = self.sample_hidden(tf.expand_dims(v, 0))
        vk, _ = self.sample_visible(hp)
        return np.array(vk)
