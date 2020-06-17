import numpy as np
import tensorflow as tf


class BoltzmannMachine:
    def __init__(self, n_visible_units, n_hidden_units, k=1, lr=1e-3):
        self.n_visible_units = n_visible_units
        self.n_hidden_units = n_hidden_units
        self.lr = lr
        self.k = k
        self.W = np.random.uniform(-0.1, 0.1, size=(self.n_visible_units, self.n_hidden_units))
        self.v_bias = np.zeros(n_visible_units)
        self.h_bias = np.zeros(n_hidden_units)

    def sample_hidden(self, v):
        probability_h_given_v = self.sigmoid(v @ self.W + self.h_bias)
        return probability_h_given_v, np.random.binomial(1, probability_h_given_v)

    def sample_visible(self, h):
        probability_v_given_h = self.sigmoid(h @ self.W.T + self.v_bias)
        return probability_v_given_h, np.random.binomial(1, probability_v_given_h)

    def update(self, v0, vk, p_h0, p_hk):
        self.W += self.lr * (v0.T @ p_h0 - vk.T @ p_hk)
        self.v_bias += self.lr * (np.sum(v0 - vk, axis=0))
        self.h_bias += self.lr * (np.sum(p_h0 - p_hk, axis=0))

    def contrastive_divergence(self, v):
        v0 = v
        vk = v.copy()
        ph_v, _ = self.sample_hidden(v0)
        for k in range(self.k):
            _, hk = self.sample_hidden(vk)
            _, vk = self.sample_visible(hk)
            vk[v0 < 0] = v0[v < 0]

        ph_k, _ = self.sample_hidden(vk)
        self.update(v0, vk, ph_v, ph_k)
        return v0, vk

    def train(self, v, n_epochs, batch_size):
        for i in range(n_epochs):
            loss = 0
            for j in range(0, len(v), batch_size):
                v_j = v[j:j + batch_size, :]
                v0, vk = self.contrastive_divergence(v_j)
                loss += np.mean(np.abs(v0[v0 >= 0] - vk[v0 >= 0]))
            print(loss)
            print(f'epoch {i}')

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
