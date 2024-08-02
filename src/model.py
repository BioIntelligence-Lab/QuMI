from typing import Callable

import jax
import optax
from flax import linen as nn
from flax import traverse_util

# import nihcxr
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.training.train_state import TrainState
from flax.typing import Dtype, Initializer
from jax import numpy as jnp
from transformers import FlaxResNetModel
from vendored import quantum


def load_resnet():
    resnet = FlaxResNetModel.from_pretrained('microsoft/resnet-50')
    module = resnet.module  # Extract the Flax Module
    variables = (
        resnet.params
    )  # Extract the parameters. Both params and batch_stats have to be transferred.
    return module, variables


# returns 0s.
class FakeClassifier(nn.Module):
    num_labels: int

    @nn.compact
    def __call__(self, x):
        return jnp.zeros((x.shape[0], self.num_labels))


# a simple linear classifier
class Classifier(nn.Module):
    backbone: nn.Module
    num_labels: int

    @nn.compact
    def __call__(self, x):
        x = self.backbone(x)
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/resnet/modeling_flax_resnet.py#L660
        x = x.pooler_output[:, :, 0, 0]  # get the correct shape of the pooled output
        return nn.Dense(features=self.num_labels, name='head')(x)


class QuantumCircuit(nn.Module):
    num_labels: int
    num_layers: int
    circuit: Callable
    param_dtype: Dtype = jnp.float32
    circuit_weights_init: Initializer = initializers.normal(2 * jnp.pi)

    # https://flax.readthedocs.io/en/latest/_modules/flax/linen/linear.html#Dense
    @nn.compact
    def __call__(self, x):
        # creates weights if they don't exist already
        circuit_weights = self.param(
            'circuit_weights',
            self.circuit_weights_init,
            (self.num_layers, self.num_labels),
            self.param_dtype,
        )  # returns a list, but we need a jax array
        circuit_weights = promote_dtype(circuit_weights, dtype=self.param_dtype)[0]

        x = jnp.tanh(x) * jnp.pi / 2  # rescale
        x = self.circuit(circuit_weights, x)  # eval
        return jnp.array(x).T  # get the right shape


class DressedQuantumClassifier(nn.Module):
    backbone: nn.Module
    num_labels: int
    num_layers: int
    circuit: Callable

    @nn.compact
    def __call__(self, x):
        x = self.backbone(x)
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/resnet/modeling_flax_resnet.py#L660
        x = x.pooler_output[:, :, 0, 0]  # get the correct shape of the pooled output
        # pass this into the quantum circuit
        x = nn.Dense(features=self.num_labels, name='input_weights')(x)
        x = QuantumCircuit(
            num_labels=self.num_labels,
            num_layers=self.num_layers,
            circuit=self.circuit,
            name='mid_weights',
        )(x)
        return nn.Dense(features=self.num_labels, name='output_weights')(x)


def create_train_step(model, params, optimizer):
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # https://huggingface.co/blog/afmck/flax-tutorial
    # must be defined this way to ensure state is not a parameter during tracing
    def predict(params, x):
        x = jax.nn.sigmoid(state.apply_fn(params, x))
        return x

    def loss_fn(params, x, y):
        # print(f'Entering loss_fn: {x.shape}, {x.dtype}, {y.shape}, {y.dtype}')
        logits = state.apply_fn(params, x)
        loss = optax.sigmoid_binary_cross_entropy(logits, y).mean()  # when sum, gradient explodes
        return loss

    def train_step(state, x, y):
        # print(f'Entering train_step: {x.shape}, {x.dtype}, {y.shape}, {y.dtype}')
        loss, grads = jax.value_and_grad(loss_fn)(state.params, x, y)
        state = state.apply_gradients(grads=grads)
        return state, loss

    return train_step, loss_fn, predict, state


def fake_model(key, num_labels=2, *args):  # noqa: ARG001
    """return a model that has correct shapes but doesn't load the actual model"""
    model = FakeClassifier(num_labels)
    # initialize with random weights
    dummy_image = jnp.empty((1, 224, 224, 3))
    params = model.init(key, dummy_image)
    optimizer = optax.set_to_zero()  # 0 optimizer
    train_step, loss_fn, predict, state = create_train_step(model, params, optimizer)
    return model, train_step, loss_fn, predict, state


def make_model(key, num_labels=2, num_layers=3, head='classical', learning_rate=1e-4, freeze=False):
    # return fake_model( key, num_labels, head, learning_rate, freeze)
    resnet, resnet_variables = load_resnet()  # extract backbone
    if head == 'classical':
        print('Building classical model')
        model = Classifier(backbone=resnet, num_labels=num_labels)
    elif head == 'quantum':
        print('Building quantum model')
        circuit = quantum.construct_circuit(
            n_layers=num_layers, n_qubits=num_labels
        )  # expensive to construct circuit inside call, so do it outside call
        circuit = jax.vmap(circuit, in_axes=(None, 0))

        model = DressedQuantumClassifier(
            backbone=resnet, num_labels=num_labels, num_layers=num_layers, circuit=circuit
        )
    else:
        msg = f'{head} is not classical or quantum'
        raise ValueError(msg)

    # initialize with random weights
    dummy_image = jnp.empty((1, 224, 224, 3))
    params = model.init(key, dummy_image)

    # transfer parameters, where params and batch_stats are toplevel keys, and the resnet variables are split into lower levels
    params['params']['backbone'] = resnet_variables['params']
    params['batch_stats']['backbone'] = resnet_variables['batch_stats']

    if freeze:
        # freeze weights by marking only the backbone as trainable
        partition_optimizers = {
            'trainable': optax.adam(learning_rate),
            'frozen': optax.set_to_zero(),
        }
        param_partitions = traverse_util.path_aware_map(
            lambda path, v: 'frozen' if 'backbone' in path else 'trainable',  # noqa: ARG005
            params,
        )
        optimizer = optax.multi_transform(partition_optimizers, param_partitions)
    else:
        # the qml benchmark paper did not freeze the weights of the transferred model
        optimizer = optax.adam(learning_rate)

    train_step, loss_fn, predict, state = create_train_step(model, params, optimizer)
    return model, train_step, loss_fn, predict, state


"""
# Testing frozen weights
# visualize a subset of the param_partitions structure
flat = list(traverse_util.flatten_dict(param_partitions).items())
print(traverse_util.unflatten_dict(dict(flat[:3] + flat[-3:])))

def test_frozen_params(params_before, params_after):
    vector_after = tm.Vector(params_after)
    vector_before = tm.Vector(params_before)
    vector_zero = (vector_before - vector_after) == 0
    # assert that all nontrainable weights are 0: flatten the tree and check that every array is 0
    assert all([arr.all() for arr in jax.tree.flatten(vector_zero.tree['params']['backbone'])[0]])
    assert all([arr.all() for arr in jax.tree.flatten(vector_zero.tree['batch_stats']['backbone'])[0]])
    # assert that at least one trainable weight is non-zero. unlikely to fail, as long as inputs are nonzero.
    assert not all([arr.all() for arr in jax.tree.flatten(vector_zero.tree['params']['head'])[0]])
"""
