# This file has been modified from the original:
# https://github.com/XanaduAI/qml-benchmarks/blob/main/src/qml_benchmarks/models/dressed_quantum_circuit.py
#
# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pennylane as qml

# TODO: the linearly separable version doesn't use entanglement and can be simulated by n * 1-qubit circuits.
# careful - jax doesn't error on invalid indices
# shape of weights should be (layers, n_qubits)


# changed: enable passing a dev so we can share one across circuits
def construct_circuit(
    dev_type='default.qubit.jax', n_layers=2, n_qubits=2, qnode_kwargs=None, dev=None
):
    if qnode_kwargs is None:
        qnode_kwargs = {'interface': 'jax-jit'}
    dev = qml.device(dev_type, wires=n_qubits) if dev is None else dev

    @qml.qnode(dev, **qnode_kwargs)
    def circuit(circuit_weights, x):
        """
        The variational circuit taken from the plots
        """
        # data encoding
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RY(x[i], wires=i)
        # trainable unitary
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(circuit_weights[layer, i], wires=i)
            qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern='ring')

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    return circuit


def test_construct_circuit():
    circuit = construct_circuit(dev_type='default.qubit', qnode_kwargs={})
    print(dir(circuit))
    import numpy as np

    print(
        qml.draw(circuit, decimals=2, expansion_strategy='device')(
            np.array([[1, 2], [3, 4]]), [5, 6]
        )
    )


def draw_circuit(n_qubits):
    import numpy as np
    import matplotlib.pyplot as plt
    circuit = construct_circuit(dev_type='default.qubit', qnode_kwargs={}, n_layers=3, n_qubits=n_qubits)

    qml.draw_mpl(circuit, expansion_strategy='device')(np.zeros((3, n_qubits)), np.arange(n_qubits))
    plt.savefig(f"circuit-drawings/{n_qubits}.svg")
    plt.close()

if __name__ == '__main__':
    # test_construct_circuit()
    draw_circuit(8)
    draw_circuit(14)
    draw_circuit(19)

