# creates and benchmarks an equivalent jax model
# benchmarks use python 3.11 for consistency

import jax
import pandas as pd
from jax import numpy as jnp
from model import make_model
from utils import benchmark, setup_jax_devices

setup_jax_devices()


def make_tensors(n_qubits, batch_size):
    # jax uses channels-first format
    inputs = jnp.zeros((batch_size, 224, 224, 3))
    labels = jnp.zeros((batch_size, n_qubits))

    return inputs, labels


def test_eval(model, inputs):
    # check that evaluation works
    outputs = model(inputs)
    print(outputs.shape)


### training loop ###


def time_train_model(state, train_step, inputs, labels, steps=1, warmup=1):
    def this_train_step():
        nonlocal state
        state, loss = jax.block_until_ready(
            train_step(state, inputs, labels)
        )  # wait for async results to finish

    return benchmark(this_train_step, rounds=steps, warmup=warmup)


def benchmark_fn(n_qubits, head, batch_size):
    print(f'n_qubits={n_qubits}, head={head}, batch_size={batch_size}')

    model, train_step, loss_fn, predict, state = make_model(
        jax.random.PRNGKey(0),
        num_labels=n_qubits,
        num_layers=3,
        head=head,
        learning_rate=1e-4,
        freeze=False,
    )
    # only keep what we need
    del model
    del predict
    del loss_fn
    inputs, labels = make_tensors(n_qubits=n_qubits, batch_size=32)
    train_step = jax.jit(train_step)
    return time_train_model(state, train_step, inputs, labels, steps=30, warmup=10)


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
    times_df.to_csv('benchmark_times/jax_model_times.csv')
    times_df_desc = times_df.T.describe()
    print(times_df_desc)
    times_df_desc.to_csv('benchmark_times/jax_model_times_desc.csv')
