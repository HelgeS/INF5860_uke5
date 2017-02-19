import numpy as np
from matplotlib import pyplot as plt




def greyscale_shift(x):

  """
  Return values for x that is invariant to changes in greyscale. X may be 1D or 2D of any (reasonably small size)

  You may need to change the number of weight-matrices sendt to neural_net_from_parameters.
  """

  W1 = np.eye(x.shape[0])
  return neural_net_from_parameters(x, [W1])


def translation_shift_1_step_size2(x):
  """
  In this exercise x will always be a vector of size 2 (2,).
  Return values that will give the same results if the two values in x is swapped,
  but a different value if a value is added to both values.

  You may need to change the number of weight-matrices sendt to neural_net_from_parameters.
  """
  W1 = np.eye(2)
  return neural_net_from_parameters(x, [W1])


def abs_function_network(x):
  """
  Build a network that return the same result as an absolutt-value function.
  Unchanged if positive and fliped sign if negative.

  You may need to change the number of weight-matrices sendt to neural_net_from_parameters.
  """
  W1 = np.eye(x.shape[0])
  return neural_net_from_parameters(x, [W1])


def multiply_network(x):
  """
  The input vector contains 2 values, both positive.
  Build the weights for a network that return the product of the two numbers.
  In this exercise we use a "non-linearity" (function between the matrix multiplication),
  that each number. See ( ... )**2

  You may need to change the number of weight-matrices sendt to neural_net_from_parameters.
  """
  W1 = np.eye(2)
  W2 = np.zeros((1, 2))
  return W2.dot( W1.dot(x)**2 )


def multiply_network_relu(x):
  """
  This exercise is the same as multiply network, but instead of using
  the squared value as "non-linearity", you will use the more standard
  function max(0, x) (Relu). This means that you can only get an approximate answer.

  To make it easy you can expect that the incoming values are integers between 0 and 11 (0<=x<11)
  PROTIP: Use bias b as well as W
  OBS: no solution on github...

  You may need to change the number of weight-matrices sendt to neural_net_from_parameters.
  """
  W1 = np.zeros((40, 2))
  b1 = np.zeros((40))
  W2 = np.zeros((40, 40))
  b2 = np.zeros((40))
  W3 = np.zeros((1, 40))
  b3 = np.zeros((1))
  return neural_net_from_parameters(x, [W1, W2, W3], [b1, b2, b3])


def neural_net_from_parameters(x, Ws, bs=None):
  """
  OBS: Do not change this file
  Calculate a neural network on the form:
  W3 max(0, W2 max(0, W1 x + b1) + b2) + b3
  :param x: the input vector to the neural network
  :param Ws: a list of weight-metrices for each layer
  :param bs: a list of bias vectors for each layer
  :return: the output of the neural network
  """

  if bs is None:
    bs = [np.zeros((w.shape[0],)) for w in Ws]
  in_x = x
  out = 0
  for w, b in zip(Ws, bs):
    out = w.dot(in_x) + b
    in_x = np.maximum(0, out)
  return out
