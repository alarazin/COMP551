#gates and circuits 

import cirq
import sympy 
import numpy as np

def one_qubit_unitary(bit, symbols):
    return cirq.Circuit(
        cirq.X(bit)**symbols[0],
        cirq.Y(bit)**symbols[1],
        cirq.Z(bit)**symbols[2])


def conv_filter_2x2(bits, symbols):
    
    circuit = cirq.Circuit()
    circuit += one_qubit_unitary(bits[0], symbols[0:3])
    circuit += one_qubit_unitary(bits[1], symbols[3:6])
    circuit += one_qubit_unitary(bits[2], symbols[6:9])
    circuit += one_qubit_unitary(bits[3], symbols[9:12])
    
    for first, second in zip(bits[0::2], bits[1::2]):
      circuit += [cirq.ZZ(first,second)**symbols[12]]
      circuit += [cirq.YY(first,second)**symbols[13]]
      circuit += [cirq.XX(first,second)**symbols[14]]

    circuit += one_qubit_unitary(bits[0], symbols[15:18])
    circuit += one_qubit_unitary(bits[1], symbols[18:21])
    circuit += one_qubit_unitary(bits[2], symbols[21:24])
    circuit += one_qubit_unitary(bits[3], symbols[24:])


    #pooling
    circuit+=cirq.CNOT(control=bits[0], target=bits[1])
    circuit+=cirq.CNOT(control=bits[2], target=bits[3])
    circuit+=cirq.CNOT(control=bits[1], target=bits[3])
    return circuit


  
def conv_circuit(bits, symbols): #all bits, sliding here. 
  circuit=cirq.Circuit()
  n=int(np.sqrt(len(bits)))
  patch=np.array(bits).reshape((n,n))
  count=0
  for j in range(0,n,2):
    for k in range(0,n,2):
      
      circuit+= conv_filter_2x2([patch[j,k], patch[j,k+1], patch[j+1,k], patch[j+1,k+1]], symbols[count:count+27])
      count+=27
  return circuit


  