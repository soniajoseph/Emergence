import numpy as np

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

def noise_image(test_image, percentage):
  """
  Args:
    test_image (nx1 array): Flattened vector.
    percentage (float): Fraction to make noisy between [0, 1]

  Returns:
    noise_image (nx1 array): Noisy image.
  """
  noise_image = test_image.copy()
  for i in range(int(len(test_image)*percentage)):
    a = np.random.choice(len(test_image))
    noise_image[a] = np.negative(noise_image[a])
  return noise_image
