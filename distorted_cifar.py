from matplotlib import pyplot as plt
import numpy as np


def normalize_images(images):
  """
  OH NOES, someone have messed up the greylevels in the cifar10 images.
  Return some representation on normalization of the images, so the
  Naive Bayes classifier gets at least 27% correct classification rate.
  """
  from skimage.filters import sobel
  a = [np.stack([sobel(i[:, :, 0]), sobel(i[:, :, 1]), sobel(i[:, :, 2])]) for i in images]
  return np.stack(a).transpose((0, 2, 3, 1))
