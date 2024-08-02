import logging
from timeit import default_timer as timer

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)

# when using manually compiled functions (through lower/compile, not through jax jit, a TypeError is raised. Use this to our advantage.
# TypeError: Argument types differ from the types for which this computation was compiled. The mismatches are:
# Argument 'x' compiled with float32[32,224,224,3] and called with float32[18,224,224,3]
# Argument 'y' compiled with bool[32,8] and called with bool[18,8]
# Benefit from https://docs.python.org/3.11/whatsnew/3.11.html#optimizations , as thanks to batching, only 1 batch at most will raise this error.

def train_loop(state, dataloader, train_step, train_step_jit):
    # logging.info('Start train loop')
    batches = iter(tqdm(dataloader, desc='train', leave=False))
    loss = 0
    for img, labels in batches:
        try:
            state, batch_loss = train_step_jit(state, img, labels)
        except TypeError:
            state, batch_loss = train_step(state, img, labels)
        loss += float(batch_loss)

    # logging.info('End train loop')
    return state, loss / len(dataloader)  # return average loss over batches. the iterator doesn't have a length somehow.


def val_loop(state, dataloader, loss_fn, loss_fn_jit):
    # logging.info('Start val loop')
    batches = iter(tqdm(dataloader, desc='val', leave=False))


    loss = 0
    for img, labels in batches:
        try:
            batch_loss = loss_fn_jit(state.params, img, labels)
        except TypeError:
            batch_loss = loss_fn(state.params, img, labels)
        loss += float(batch_loss)

    # logging.info('End val loop')
    return loss / len(dataloader)


def test_loop(state, dataloader, loss_fn, loss_fn_jit):
    # logging.info('Start test loop')
    batches = iter(tqdm(dataloader, desc='test', leave=False))

    loss = 0
    for img, labels in batches:
        try:
            batch_loss = loss_fn_jit(state.params, img, labels)
        except TypeError:
            batch_loss = loss_fn(state.params, img, labels)
        loss += float(batch_loss)

    # logging.info('End test loop')
    return loss / len(dataloader)


def predict_loop(state, dataloader, predict, predict_jit):
    # logging.info('Start predict')

    batches = iter(tqdm(dataloader, desc='predict', leave=False))

    preds = []
    for img, labels in batches:
        try:
            batch_preds = predict_jit(state.params, img)
        except TypeError:
            batch_preds = predict(state.params, img)
        preds.append(np.asarray(batch_preds))

    # logging.info('End predict')
    return np.concatenate(preds)


def epoch_loop(
    state,
    epochs,
    patience,
    checkpointer,
    train_loader,
    val_loader,
    train_step,
    loss_fn,
    train_step_jit,
    loss_fn_jit,
):
    """
    Your garden-variety training loop, with early stopping and checkpointing.

    Not strictly a loop, but to move this into a separate loop, a dictionary would have to be used to manage state.
    Plus it would make handling history and state more complicated.
    """

    logging.info('Starting epochs')
    attempts = 0
    best_val_loss = float('inf')

    # TODO: move into a metrics object?
    train_losses = []
    val_losses = []
    train_times = []
    val_times = []

    for epoch in (pbar := trange(epochs, desc='Epoch', leave=False)):
        logging.info('Start epoch: %s', epoch)

        logging.info('Start train loop')
        train_time = timer()
        state, train_loss = train_loop(state, train_loader, train_step, train_step_jit)
        # not ergonomic to always surround functions with time, but writing a decorator is also not ergonomic
        train_time = timer() - train_time
        logging.info('End train loop')
        train_times.append(train_time)
        train_losses.append(train_loss)

        logging.info('Start val loop')
        val_time = timer()
        val_loss = val_loop(state, val_loader, loss_fn, loss_fn_jit)

        val_time = timer() - val_time
        logging.info('End val loop')
        val_times.append(val_time)
        val_losses.append(val_loss)

        logging.info('End epoch: %s', epoch)

        # checkpoint and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logging.info('New best val loss: %.3f', val_loss)
            logging.info('Saving new checkpoint for epoch %s', epoch)
            pbar.set_postfix({'Best epoch': epoch, 'Best val loss': f'{best_val_loss:.3f}'})
            checkpointer.save(state)
            attempts = 0
        else:
            attempts += 1
            logging.info('Losing patience: %s attempts without improvement so far', attempts)
            if attempts >= patience:
                logging.info('Ran out of patience! Stopping early')
                break

    logging.info('Finished training after %s epochs!', epoch + 1)
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_time': train_time,
        'val_time': val_time,
    }

    return state, history
