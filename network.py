#models
import sympy
import cirq
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq



#encode images to quantum states

def encode_circuit(values):
  im4=values[1:5,1:5]
  phi=np.ndarray.flatten(im4)
  circuit=cirq.Circuit()
  qubits=cirq.GridQubit.rect(4,4)
  for i in range(16):
    circuit.append(cirq.ry(np.pi*phi[i])(qubits[i]))
  return circuit

"""
def create_model(qubits):
  model_circuit=cirq.Circuit()
  symbols=sympy.symbols('qconv0:200')
  model_circuit+=conv_circuit(qubits, symbols[:108])
  index=[5,7,13,15]
  model_circuit+=conv_circuit([qubits[i] for i in index], symbols[108:])
  return model_circuit
"""
#qubits16=cirq.GridQubit.rect(4,4)
#readout_operators=cirq.Z(qubits16[-1])

#model=tf.keras.Sequential([tf.keras.layers.Input(shape=(), dtype=tf.string), 
#                           tfq.layers.PQC(create_model(qubits16), readout_operators)])

class QCNN_model:
  def __init__(self, qubits, conv_layer):
    self.qubits = qubits
    self.readout_operator = cirq.Z(self.qubits[-1])
    self.conv_circuit = conv_layer

  def create_model_circuit(self):
    self.model_circuit=cirq.Circuit()
    symbols=sympy.symbols('qconv0:200')
    self.model_circuit+=self.conv_circuit(self.qubits, symbols[:108])
    index=[5,7,13,15]
    self.model_circuit+=self.conv_circuit([self.qubits[i] for i in index], symbols[108:])
    return self.model_circuit

  def build_network(self):
    self.model = tf.keras.Sequential([tf.keras.layers.Input(shape=(), dtype=tf.string), tfq.layers.PQC(self.create_model_circuit(), self.readout_operator)])
    return self.model






  