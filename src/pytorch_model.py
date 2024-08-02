# creates and benchmarks an equivalent pytorch model
# benchmarks use python 3.11 for consistency

import numpy as np
import pandas as pd
import pennylane as qml
import torch
from torch import nn
from torchvision import models
from utils import benchmark

# https://discuss.pennylane.ai/t/runtimeerror-with-mixed-device-tensors-when-integrating-pennylane-with-pytorch/4318/4
# imposes slight performance cost: https://pytorch.org/docs/stable/generated/torch.set_default_device.html

torch.set_default_device('cuda')
# faster matrix mults
torch.set_float32_matmul_precision('high')


# copied from vendored, except indicate that batching is allowed
# https://discuss.pennylane.ai/t/question-about-torchlayer/4088/6
def construct_circuit(
    dev_type='default.qubit', n_layers=2, n_qubits=2, qnode_kwargs=None, dev=None
):
    dev = qml.device(dev_type, wires=n_qubits) if dev is None else dev

    # avoid broadcasting overhead
    # https://docs.pennylane.ai/en/stable/introduction/templates.html#broadcasting-function
    @qml.qnode(dev)
    def circuit(inputs, weights):
        """
        The variational circuit taken from the plots, adjusting to allow for batching
        """
        # initialize to 50/50 0/1
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        # data encoding after rescaling
        qml.AngleEmbedding(inputs * 2 * np.pi, wires=range(n_qubits), rotation='Y')
        # trainable unitary
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=qml.RY)

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    return circuit


def make_model(n_layers=3, n_qubits=2, head='classical', dev_type=None):
    model = models.resnet50(weights='DEFAULT')

    weight_shapes = {'weights': (n_layers, n_qubits)}

    if head == 'quantum':
        # create quantum circuit
        circuit = construct_circuit(n_layers=n_layers, n_qubits=n_qubits)
        circuit_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        layers = [
            nn.Tanh(),
            nn.Linear(model.fc.in_features, n_qubits),
            circuit_layer,
            nn.Linear(n_qubits, n_qubits),
        ]

    else:
        layers = [nn.Linear(model.fc.in_features, n_qubits)]

    model.fc = nn.Sequential(*layers)
    return model, weight_shapes


def make_tensors(n_qubits, batch_size):
    # pytorch uses channels-first format
    inputs = torch.zeros((batch_size, 3, 224, 224))
    labels = torch.zeros((batch_size, n_qubits))

    return inputs, labels


def test_eval(model, inputs):
    # check that evaluation works
    model.eval()
    outputs = model(inputs)
    print(outputs.shape)


### training loop ###


def time_train_model(model, inputs, labels, steps=1, warmup=1):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()

    # cannot compile until this is solved:
    # https://github.com/pytorch/pytorch/issues/93624
    # @torch.compile
    def train_step():
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    return benchmark(train_step, rounds=steps, warmup=warmup)


def benchmark_fn(n_qubits, head, batch_size):
    print(f'n_qubits={n_qubits}, head={head}, batch_size={batch_size}')

    model, weight_shapes = make_model(n_qubits=n_qubits, n_layers=3, head=head)
    inputs, labels = make_tensors(n_qubits=n_qubits, batch_size=32)
    return time_train_model(model, inputs, labels, steps=30, warmup=10)


if __name__ == '__main__':
    results = []
    batch_size = 32
    n_qubits_lst = [8, 14, 19]
    head_lst = ['quantum', 'classical']
    for n_qubits in n_qubits_lst:
        for head in head_lst:
            times = benchmark_fn(n_qubits, head, batch_size)
            results.append(times)
    times_df = pd.DataFrame(
        results,
        index=pd.MultiIndex.from_product(
            [n_qubits_lst, head_lst], names=['n_labels', 'classifier']
        ),
    )
    times_df.to_csv('benchmark_times/pytorch_model_times.csv')
    times_df_desc = times_df.T.describe()
    print(times_df_desc)
    times_df_desc.to_csv('benchmark_times/pytorch_model_times_desc.csv')
