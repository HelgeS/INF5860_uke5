from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
from cifar import load_cifar_file
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

from distorted_cifar import normalize_images


def test_distorted_cifar():
  images, labels = load_cifar_file('./input/data_batch_1')
  images = normalize_images(images)
  clf = GaussianNB()
  validation_score = cross_val_score(clf, images.reshape((images.shape[0], -1)), labels, cv=10).mean()
  print('validation_score', validation_score)
  assert validation_score > 0.27