# setup_jax_devices('auto')

import json
import logging
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
import jax.random as jrand
import pandas as pd
from dataloader import make_dataloader
from loops import epoch_loop, predict_loop, test_loop

# from model import load_model
from vendored.spinner import Spinner


def compile_and_log_cost(save_path, fn, *args):
    """
    Measure compile time and save cost analysis.
    Note: 'JAX cannot guarantee that the output of compiled.cost_analysis() on one day will remain the same on the following day.'
    Ref: https://jax.readthedocs.io/en/latest/aot.html#debug-information-and-analyses-when-available
    Ref: https://github.com/google-research/scenic/blob/421391cd67faffb5e2de7898d67a6d71fa5ec3db/scenic/common_lib/debug_utils.py#L106
    Note that FLOPS = 2*MAC (multiply/add)
    """
    compile_time = timer()
    jitted = jax.jit(fn)
    lowered = jitted.lower(*args)
    compiled = lowered.compile()
    compile_time = timer() - compile_time
    with open(str(save_path) + '_compile_time.txt', 'w') as fp:
        fp.write(f'compile_time\n{compile_time}\n')

    cost_lowered = lowered.cost_analysis()
    cost_compiled = compiled.cost_analysis()[0]

    with open(str(save_path) + '_lowered.json', 'w') as fp:
        json.dump(dict(sorted(cost_lowered.items())), fp)
    with open(str(save_path) + '_compiled.json', 'w') as fp:
        json.dump(dict(sorted(cost_compiled.items())), fp)

    logging.info('Saved cost analyses')
    return compiled  # return the compiled function only


def train(cfg, model_objects):
    # unpack model objects
    cache = model_objects['cache']
    checkpointer = model_objects['checkpointer']
    # model = model_objects['model']
    train_step = model_objects['train_step']
    loss_fn = model_objects['loss_fn']
    # predict = model_objects['predict']
    state = model_objects['state']
    key = model_objects['key']
    model_objects['key'], train_key, val_key = jrand.split(key, num=3)

    # unpack cfg objects
    batch_size = cfg['batch_size']
    num_labels = cfg['num_labels']
    threads = cfg['threads']
    fraction = cfg['fraction']
    epochs = cfg['epochs']
    patience = cfg['patience']
    jit = cfg['jit']

    if jit:
        with Spinner('Warming up JIT...'):
            warmup_images = jnp.ones(
                (batch_size, 224, 224, 3), dtype=jnp.float32
            )  # TODO: use the same dtype as cached image
            warmup_labels = jnp.ones((batch_size, num_labels), dtype=bool)
            train_step_jit = compile_and_log_cost(
                cfg['cost_dir'] / 'train_fn_cost', train_step, state, warmup_images, warmup_labels
            )
            loss_fn_jit = compile_and_log_cost(
                cfg['cost_dir'] / 'loss_fn_cost',
                loss_fn,
                state.params,
                warmup_images,
                warmup_labels,
            )
            del warmup_images
            del warmup_labels
    else:
        train_step_jit = train_step
        loss_fn_jit = loss_fn

    model_objects['loss_fn'] = loss_fn  # save for train step
    model_objects['loss_fn_jit'] = loss_fn_jit

    train_loader = make_dataloader(
        'train',
        frac=fraction,
        threads=threads,
        batch_size=batch_size,
        cache=cache,
        labels=num_labels,
        key=train_key,
        shuffle=True,
        augment=True,
    )
    val_loader = make_dataloader(
        'val',
        frac=fraction,
        threads=threads,
        batch_size=batch_size,
        cache=cache,
        key=val_key,
        labels=num_labels,
    )

    print('Train phase start')
    epoch_time = timer()
    state, history = epoch_loop(
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
    )
    epoch_time = timer() - epoch_time
    print('Train phase end (finally!)')

    # cleanup stored train step function, it won't be used for testing
    # the jitted version will go out of scope when this function exits
    del model_objects['train_step']
    model_objects['state'] = state  # save state for testing

    logging.info('Save history')
    hist_file = cfg['results_dir'] / 'history.csv'
    hist_df = pd.DataFrame(history)
    hist_df.index.name = 'epoch'
    hist_df.to_csv(hist_file)

    with open(cfg['results_dir'] / 'epoch_time.txt', 'w') as f:
        f.write(f'epoch time\t{epoch_time}\n')

    checkpointer.checkpointer.wait_until_finished()
    return model_objects


def test(cfg, model_objects):
    cache = model_objects['cache']
    checkpointer = model_objects['checkpointer']
    # model = model_objects['model']
    # train_step = model_objects['train_step']
    loss_fn = model_objects['loss_fn']
    predict = model_objects['predict']
    state = model_objects['state']
    key = model_objects['key']

    # unpack cfg objects
    batch_size = cfg['batch_size']
    num_labels = cfg['num_labels']
    threads = cfg['threads']
    fraction = cfg['fraction']
    jit = cfg['jit']

    print('Loading best checkpoint')
    logging.info('Loading best checkpoint')
    best_state = checkpointer.restore(state)

    test_loader = make_dataloader(
        'test',
        frac=1,
        threads=threads,
        batch_size=batch_size,
        cache=cache,
        labels=num_labels,
        key=key,
    )

    if jit:
        with Spinner('Warming up JIT...'):
            warmup_images = jnp.ones((batch_size, 224, 224, 3), dtype=jnp.float32)
            predict_jit = compile_and_log_cost(
                cfg['cost_dir'] / 'predict_cost.json', predict, state.params, warmup_images
            )
        if 'loss_fn_jit' in model_objects:  # restore it
            loss_fn_jit = model_objects['loss_fn_jit']
        else:  # recompile
            warmup_labels = jnp.ones((batch_size, num_labels), dtype=bool)
            loss_fn_jit = compile_and_log_cost(
                cfg['cost_dir'] / 'loss_fn_cost',
                loss_fn,
                state.params,
                warmup_images,
                warmup_labels,
            )
        del warmup_images
        del warmup_labels
    else:
        predict_jit = predict
        loss_fn_jit = loss_fn

    print('Test phase start')
    print('Test loss')
    logging.info('Start test loop')
    test_time = timer()
    test_loss = test_loop(best_state, test_loader, loss_fn, loss_fn_jit)
    test_time = timer() - test_time
    logging.info('End test loop')

    print('Get test predictions')
    logging.info('Start predict')
    pred_time = timer()
    preds = predict_loop(best_state, test_loader, predict, predict_jit)
    pred_time = timer() - pred_time
    logging.info('End predict')

    logging.info('Save test results')
    # only three items here, open directly
    with open(cfg['results_dir'] / 'test_loss_time.txt', 'w') as f:
        f.write(f'test_loss\t{test_loss}\n')
        f.write(f'test_time\t{test_time}\n')
        f.write(f'pred_time\t{pred_time}\n')

    logging.info('Save test predictions')
    preds_df = pd.DataFrame(preds, index=test_loader.image_paths, columns=test_loader.label_cols)
    preds_file = cfg['results_dir'] / 'preds.csv'
    preds_df.to_csv(preds_file)

    print('Test phase end')
    print(f'Results: {cfg["results_dir"]}')
