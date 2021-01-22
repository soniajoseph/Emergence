import numpy as np
import matplotlib.pyplot as plt 

class Hopfield():

  def __init__(self):
    self.T = None

  def run(self, V_noise, iterations):
    """
    Run network.

    Args:
      V_prime (nx1 array): Flattened noisy vector.
      T (nxn array): Weight matrix.
      time (int): Number of iterations.
    Returns:
      V_hat (nx1 array): Updated V_prime vector.
    """
    num_neurons = len(self.T)
    U = 0
    V_hat = V_noise.copy()
    for t in range(iterations):
      e = self.energy(V_hat)
      for i, _ in enumerate(self.T[:,0]):
        total = 0
        for j,_ in enumerate(self.T[i,:]):
          if j != i:
            total += self.T[i][j] * V_hat[j]
        if total > U:
          V_hat[i] = 1
        else:
          V_hat[i] = -1
      e_new = self.energy(V_hat)
      if e_new == e:
        return V_hat
    return V_hat


  def energy(self, v):
    """
    Get energy function of network, useful for knowing when network has converged.

    Args:
      v (vector)

    Returns:
      e (float)
    """
    e = -.5 * v @ self.T @ v 
    return e


  def train(self, data):
    """
    Train network.

    Args:
      data (list of nx1 arrays): List of flattened arrays to store in network.

    Returns:
      T (nxn array): Weight matrix.
    """
    self.T = np.zeros((len(data[0]), len(data[0])))  
    for V_s in data:
      V_s = np.array(V_s)
      self.T += np.outer(V_s, V_s)
      np.fill_diagonal(self.T, 0)
    self.T = self.T / self.T.shape[0]
    return self.T




