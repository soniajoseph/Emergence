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