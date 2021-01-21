

import collections 
import tensorflow as tf 
import numpy as np 

def filter_numbers(x,y, num1, num2):
  keep = (y==num1)| (y==num2)
  x, y = x[keep], y[keep]
  y = y == num1
  return x,y 

def convert_label(y):
  if y==True:
    return 1.0
  else:
    return -1.0

def import_data(num1,num2, size, N_train=None, N_test=None):
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

  x_train_, y_train_ = filter_numbers(x_train, y_train, num1, num2)
  x_test_, y_test_ = filter_numbers(x_test, y_test, num1, num2)

  x_train_small = tf.image.resize(x_train_, (size,size)).numpy()
  x_test_small = tf.image.resize(x_test_, (size,size)).numpy()

  y_train_ = [convert_label(y) for y in y_train_]
  y_test_ = [convert_label(y) for y in y_test_]

  if N_train==None:
    X_train = x_train_small[:]
    X_test = x_test_small[:]
    Y_train = y_train_[:]
    Y_test = y_test_[:]
  else:
    X_train = x_train_small[:N_train]
    X_test = x_test_small[:N_test]
    Y_train = y_train_[:N_train]
    Y_test = y_test_[:N_test]

  return X_train, X_test, Y_train, Y_test
