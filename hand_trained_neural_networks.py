import numpy as np
from matplotlib import pyplot as plt


def neural_net_from_parameters(x, Ws, bs=None):
  if bs is None:
    bs = [np.zeros((w.shape[0],)) for w in Ws]
  in_x = x
  out = 0
  for w, b in zip(Ws, bs):
    out = w.dot(in_x) + b
    in_x = np.maximum(0, out)
  return out


def greyscale_shift(x):
  """
  Return values for x that is invariant to changes in greyscale. X may be 1D or 2D of any (reasonably small size)
  """
  eye = np.eye(x.shape[0])
  W = -eye
  W[:, 1:] += eye[:, :-1]
  W[-1, 0] = 1
  return neural_net_from_parameters(x, [W])


def translation_shift_1_step_size2(x):
  """
  In this exercise x will always be a vector of size 2 (2,).
  Return values that will give the same results if the two values in x is swapped,
  but a different value if a value is added to both values.
  """
  W1 = np.array([[1, -1], [0, 1]])
  W2 = np.array([[1, 1], [1, 1]])
  return neural_net_from_parameters(x, [W1, W2])


def abs_function_network(x):
  """
  Build a network that return the same result as an absolutt-value function.
  Unchanged if positive and fliped sign if negative.
  """
  N = x.shape[0]
  W1 = np.zeros((N * 2, N))
  W1[::2] = np.eye(N)
  W1[1::2] = -W1[::2]
  W2 = np.abs(W1).T

  return neural_net_from_parameters(x, [W1, W2])


def multiply_network(x):
  """
  The input vector contains 2 values, both positive.
  Build the weights for a network that return the product of the two numbers.
  In this exercise we use a "non-linearity" (function between the matrix multiplication),
  that each number. See ( ... )**2
  """
  W1 = np.array([[1, 1], [1, 0], [0, 1]])
  W2 = np.array([[0.5, -0.5, -0.5]])
  return W2.dot( W1.dot(x)**2 )


def multiply_network_relu(x):
  """
  This exercise is the same as multiply network, but instead of using
  the squared value as "non-linearity", you will use the more standard
  function max(0, x) (Relu). This means that you can only get an approximate answer.

  To make it easy you can expect that the incoming values are integers between 0 and 11 (0<=x<11)
  PROTIP: Use bias b as well as W
  OBS: no solution on github...
  """
  W1 = np.zeros((40, 2))
  b1 = np.zeros((40))
  b2 = np.zeros((40))
  W2 = np.zeros((40, 2))
  W3 = np.zeros((1, 40))
  b3 = np.zeros((1))
  return neural_net_from_parameters(x, [W1, W2, W3], [b1, b2, b3])

