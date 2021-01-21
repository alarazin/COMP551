import tensorflow as tf 
import tensorflow_quantum as tfq
import cirq 
import numpy as np 
from network import encode_circuit

def prep_data(X_train,X_test):
  x_train_16=[encode_circuit(x) for x in X_train]
  x_test_16=[encode_circuit(x) for x in X_test]

  xx_train_16=tfq.convert_to_tensor(x_train_16)
  xx_test_16=tfq.convert_to_tensor(x_test_16)
  return xx_train_16, xx_test_16

def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)

def compile_model(model):
  model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=tf.losses.Hinge(),
                   metrics=[hinge_accuracy])
  return model

def train_model(model,X_train, Y_train, X_test, Y_test,epochs=25, batch_size=16, verbose=1):
  xx_train_16, xx_test_16 = prep_data(X_train,X_test)
  model_compiled = compile_model(model)
  history = model_compiled.fit(x=xx_train_16,
                         y=np.asarray(Y_train),
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=verbose,
                         validation_data=(xx_test_16, np.asarray(Y_test)))

  return model_compiled, history 

