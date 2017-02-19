import numpy as np

from hand_trained_neural_networks import greyscale_shift, translation_shift_1_step_size2, abs_function_network, multiply_network, \
  multiply_network_relu


def transform_add(x):
  return x+np.random.randint(1, 20)

def transform_shift(n):
  return lambda x: np.roll(x, n)

def transform_contrast(x):
  return x*np.random.randint(1, 20)

def test_grayscale_shift():
  for i in range(10):
    x = np.random.randint(0, 10000, size=(6, ))*100
    x1 = greyscale_shift(x)
    x2 = greyscale_shift(transform_add(x))
    assert (x1 == x2).all()
    for transform in [transform_shift(1), transform_shift(2), transform_contrast]:
      assert (x1 != transform(x)).any()

def test_translation_shift_1_step_size2():
  for i in range(10):
    x = np.random.randint(0, 255, size=(2, ))
    x[1] += 255 # make sure they are not the same value
    x1 = translation_shift_1_step_size2(x)
    x2 = translation_shift_1_step_size2(transform_shift(1)(x))
    assert (x1 == x2).all()
    for transform in [transform_contrast, transform_add]:
      assert (x1 != transform(x)).any()


def test_abs_function_network():
  for i in range(10):
    x = np.random.random((10,))*10 - 5
    x1 = abs_function_network(x)
    assert (np.abs(x) == x1).all()


def test_multiply_network():
  for i in range(10):
    x = np.random.random((2,))*10
    x1 = multiply_network(x)
    assert np.abs(np.prod(x) - x1) < 1e-12

def test_multiply_network_relu():
  for i in range(10):
    x = np.random.random((2,))*10
    x1 = multiply_network_relu(x)
    assert np.abs(np.prod(x) - x1) < 1e-12


if __name__ == '__main__':
  test_grayscale_shift()
  test_abs_function_network()
  test_translation_shift_1_step_size2()
  test_multiply_network()
  test_multiply_network_relu()