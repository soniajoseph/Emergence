import numpy as np
import tensorflow as tf

from hopfield.hopfield import Hopfield


class BoltzmannMachine(Hopfield):
    def __init__(self, n_visible_units, n_hidden_units, k=10, lr=1, n_epochs=10, batch_size=1):
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
        probability_h_given_v = self.sigmoid(v @ self.T + self.h_bias)
        return probability_h_given_v, np.random.binomial(1, probability_h_given_v)

    def sample_visible(self, h):
        probability_v_given_h = self.sigmoid(h @ self.T.T + self.v_bias)
        return probability_v_given_h, np.random.binomial(1, probability_v_given_h)

    def update(self, v0, vk, p_h0, p_hk):
        self.T += self.lr * (v0.T @ p_h0 - vk.T @ p_hk)
        self.v_bias += self.lr * (np.sum(v0 - vk, axis=0))
        self.h_bias += self.lr * (np.sum(p_h0 - p_hk, axis=0))

    def contrastive_divergence(self, v):
        v0 = v
        vk = v.copy()
        ph_v, _ = self.sample_hidden(v0)

        # Gibbs sampling
        for k in range(self.k):
            _, hk = self.sample_hidden(vk)
            _, vk = self.sample_visible(hk)
            vk[v0 < 0] = v0[v0 < 0]

        ph_k, _ = self.sample_hidden(vk)
        self.update(v0, vk, ph_v, ph_k)
        return v0, vk

    def energy(self, v):
        _, hs = self.sample_hidden(v)
        e = - v @ self.v_bias - hs @ self.h_bias - v @ self.T @ hs.T
        return e

    def train(self, data):
        for i in range(self.n_epochs):
            loss = 0
            for j in range(0, len(data), self.batch_size):
                v_j = data[j:j + self.batch_size, :]
                v0, vk = self.contrastive_divergence(v_j)
                loss += np.mean(np.abs(self.energy(v0)[0][0] - self.energy(vk)[0][0]))
            print(f'epoch {i} - loss: {loss}')

    def inference(self, v):
        hp, _ = self.sample_hidden(v)
        vk, _ = self.sample_visible(hp)
        return vk

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
