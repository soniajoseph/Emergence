import numpy as np
import tensorflow as tf

from hopfield.hopfield import Hopfield


class BoltzmannMachine(Hopfield):
    def __init__(self, n_visible_units, n_hidden_units, k=5, lr=1, n_epochs=10, batch_size=1):
        """
        Boltzmann machine network
        Args:
            n_visible_units (int): Number of visible units
            n_hidden_units (int): Number of hidden units
            k (int): k-folds for Gibbs sampling
            lr (int): learning rate
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
        self.T = np.random.uniform(-0.1, 0.1, size=(self.n_visible_units, self.n_hidden_units))
        self.v_bias = np.zeros(n_visible_units)
        self.h_bias = np.zeros(n_hidden_units)

    def sample_hidden(self, v):
        """
        Visible units binomial sampling
        Args:
            v (nd.array): input data

        Returns:
            nd.array, nd.array
        """
        probability_h_given_v = self.sigmoid(v @ self.T + self.h_bias)
        return probability_h_given_v, np.random.binomial(1, probability_h_given_v)

    def sample_visible(self, h):
        """
        Hidden units binomial sampling
        Args:
            h (nd.array): hidden units data

        Returns:
            nd.array, nd.array
        """
        probability_v_given_h = self.sigmoid(h @ self.T.T + self.v_bias)
        return probability_v_given_h, np.random.binomial(1, probability_v_given_h)

    def update(self, v0, vk, p_h0, p_hk):
        """
        Update weights and biases
        Args:
            v0 (nd.array): Original input data
            vk (nd.array): Sampled input data
            p_h0 (nd.array): Probability of h = 1 given v0
            p_hk (nd.array): Probability of h = 1 given vk

        """
        self.T += self.lr * (v0.T @ p_h0 - vk.T @ p_hk) / self.batch_size
        self.v_bias += self.lr * (np.mean(v0 - vk, axis=0))
        self.h_bias += self.lr * (np.mean(p_h0 - p_hk, axis=0))

    def contrastive_divergence(self, v):
        """
        Contrastive divergence
        Args:
            v (nd.array): input data

        Returns:
            nd.array, nd.array: original input data and sampled data
        """
        vk = v.copy()
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
            v (nd.array): input data

        Returns:
            float: energy
        """
        _, hs = self.sample_hidden(v)
        e = - v @ self.v_bias - hs @ self.h_bias - v @ self.T @ hs.T
        return e

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
                v0, vk = self.contrastive_divergence(v_j)
                loss += np.mean(np.abs(self.energy(v0)[0][0] - self.energy(vk)[0][0]))
            print(f'epoch {i} - loss: {loss}')

    def inference(self, v):
        """
        Inference
        Args:
            v (nd.array): input data

        Returns:
            nd.array: sampled output
        """
        hp, _ = self.sample_hidden(v)
        vk, _ = self.sample_visible(hp)
        return vk

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
