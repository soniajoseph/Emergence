import numpy as np

def run(V_prime, T, time):
  """
  Run network.

  Args:
    V_prime (nx1 array): Flattened noisy vector.
    T (nxn array): Weight matrix.
    time (int): Number of iterations.
  Returns:
    V_hat (nx1 array): Updated V_prime vector.
  """
  num_neurons = len(T)
  U = 0
  V_hat = V_prime.copy()
  for t in range(time):
    for i, _ in enumerate(T[:,0]):
      total = 0
      for j,_ in enumerate(T[i,:]):
        if j != i:
          total += T[i][j] * V_hat[j]
      if total > U:
        V_hat[i] = 1
      else:
        V_hat[i] = -1
  return V_hat


def train(data):
  """
  Train network.

  Args:
    data (list of nx1 arrays): List of flattened arrays to store in network.

  Returns:
    T (nxn array): Weight matrix.
  """
  T = np.zeros((len(data[0]), len(data[0])))  
  for V_s in data:
    V_s = np.array(V_s)
    T += np.outer(V_s, V_s)
    np.fill_diagonal(T, 0)
    plt.matshow(T)
  T = T / T.shape[0]
  return T


def normalize_binarize(test_image):
  """
  Normalize and binarize test_image.
  """
  test_image = test_image / 255
  mean = np.mean(test_image)
  idx_over = test_image > mean 
  idx_under = test_image <= mean 
  test_image[idx_over] = 1
  test_image[idx_under] = -1
  return test_image

