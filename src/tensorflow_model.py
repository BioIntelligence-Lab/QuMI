# creates and benchmarks an equivalent tensorflow model
# benchmarks use python 3.11 for consistency

import keras
import numpy as np
import pandas as pd
import pennylane as qml
import tensorflow as tf
from keras import activations, layers
from utils import benchmark

# torch.set_default_device('cuda')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPU')
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# hide warnings about casting from complex128 to float32
tf.get_logger().setLevel('ERROR')


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
    model = keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
    )
    seq = [
        model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(units=n_qubits),
    ]

    weight_shapes = {'weights': (n_layers, n_qubits)}

    if head == 'quantum':
        # create quantum circuit
        circuit = construct_circuit(n_layers=n_layers, n_qubits=n_qubits)
        circuit_layer = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=n_qubits)

        seq.extend(
            [
                layers.Activation(activations.tanh),
                circuit_layer,
                layers.Dense(units=n_qubits),
            ]
        )

    model = keras.Sequential(seq)
    return model, weight_shapes


def make_tensors(n_qubits, batch_size):
    # tensorflow uses channels-last format
    inputs = tf.zeros((batch_size, 224, 224, 3))
    labels = tf.zeros((batch_size, n_qubits))

    return inputs, labels


def test_eval(model, inputs):
    # check that evaluation works
    outputs = model(inputs, training=True)
    print(outputs.shape)


### training loop ###


def time_train_model(model, inputs, labels, steps=1, warmup=1):
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    # take some time to compile, but will ensure that memory usage doesn't explode
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss_value = loss_fn(labels, outputs)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights, strict=False))

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
    times_df.to_csv('benchmark_times/tensorflow_model_times.csv')
    times_df_desc = times_df.T.describe()
    print(times_df_desc)
    times_df_desc.to_csv('benchmark_times/tensorflow_model_times_desc.csv')
