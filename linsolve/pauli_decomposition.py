import numpy as np
import sys
from qiskit.opflow import (
    Z,
    X,
    Y,
    I,
    Plus,
    Minus,
    Zero, One,
    CX, S, H, T, CZ, Swap,
    TensoredOp
)
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator, Pauli
from itertools import product 

zero_op = (I + Z) / 2
one_op = (I - Z) / 2



class PauliDecomposition:

    def __init__(self, matrix):

        self._validate_matrix(matrix)
        self.nqubit = self._compute_circuit_size(matrix)
        self.matrix = self._process_matrix(matrix, self.nqubit)
        
        

        self.paulis = self.decompose()


    @staticmethod
    def _process_matrix(matrix, nqubit):

        mat_size = matrix.shape[0]
        if  mat_size == 2**nqubit:
            return matrix 
        else:
            out = np.eye(2**nqubit)
            out[:mat_size,:mat_size] = matrix
            return out

    @staticmethod
    def _validate_matrix(mat):
        shape = mat.shape
        assert len(shape)==2
        assert np.allclose(mat, np.conj(mat.T))
    
    @staticmethod
    def _compute_circuit_size(matrix):
        size = matrix.shape[0]
        return int(np.ceil(np.log2(size)))

    def decompose(self):
        decomp = dict()
        basis = 'IXYZ'
        prefactor = 1./(2**self.nqubit)
        for pauli_gates in product(basis, repeat=self.nqubit):
            paulis = ''.join(pauli_gates)
            op = Operator(Pauli(paulis)).data
            coef = np.trace(op@self.matrix)
            if coef*np.conj(coef) != 0:
                decomp[paulis] = prefactor * coef
                # print(prefactor*coef, paulis)
        return decomp

    def recompose(self):
        size = 2**self.nqubit
        mat = np.zeros((size,size)).astype('complex128')
        for paulis, coeff in self.paulis.items():
            mat +=  coeff * Operator(Pauli(paulis)).data
        return mat

    def factorize(self):

        fact = dict()
        for k,v in self.paulis.items():
            if v not in fact:
                fact[v] = k 
            else:
                fact[v] += '+' + k

        return fact

if __name__ == "__main__":

    A = np.random.rand(6,6)
    A = A + A.T

    paulis  = PauliDecomposition(A)
    paulis.recompose()