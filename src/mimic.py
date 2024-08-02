import logging
from dataclasses import dataclass
from timeit import default_timer as timer
from vendored.spinner import Spinner
import jax.numpy as jnp
import jax

import pandas as pd

import nihcxr

import trainer

from loops import test_loop, predict_loop

pd.set_option('mode.copy_on_write', True)

# use the same labels, except Effusion is known as Pleural Effusion

def labels_nihcxr_to_mimic(labels):
    return [label if label != 'Effusion' else 'Pleural Effusion' for label in labels]

labels_8 = labels_nihcxr_to_mimic(nihcxr.get_labels(8))
labels_14 = labels_nihcxr_to_mimic(nihcxr.get_labels(14))
labels_19 = labels_nihcxr_to_mimic(nihcxr.get_labels(19))
labels = labels_19

def get_labels(n):
    match n:
        case 8:
            return labels_8
        case 14:
            return labels_14
        case 19:
            return labels_19
        case _:
            msg = f'{n} labels not built in, specify labels as list instead'
            raise ValueError(msg)

def load_csv(path):
    labels_df = pd.read_csv(
        path,
        index_col='path',
        usecols=labels + ['path'],
        dtype=dict.fromkeys(labels, bool) | {'path': 'string'},
        engine='pyarrow',
    )
    labels_df.index = '/srv/store/Data/MIMIC-CXR-JPG/' + labels_df.index
    return labels_df

# not using metadata for this study

@dataclass
class Dataset:
    """Class for loading dataframes only on demand"""

    test_csv_path: str
    test_df = None

    def get_df(self):
            if self.test_df is None:
                self.test_df = load_csv(self.test_csv_path)
            return self.test_df

dataset = Dataset(
    test_csv_path='/srv/store/Projects/schan/quantum/mimic/test_clean.csv'
)

def make_mimic_dataloader(
    batch_size=4,
    cache=None,
    labels=19,
    threads=0,
):
    from dataloader import DataLoaderJax

    label_cols = get_labels(labels) if isinstance(labels, int) else labels

    dataloader_df = dataset.get_df()
    paths = dataloader_df.index
    labels = dataloader_df[label_cols]

    return DataLoaderJax(
        image_paths=paths,
        labels=labels,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        rng=None,
        cache=cache,
        label_cols=label_cols,
        threads=threads,
    )

def external_test(cfg, model_objects):

    logger = logging.getLogger(__name__)
    cache = model_objects['cache']
    checkpointer = model_objects['checkpointer']
    # model = model_objects['model']
    # train_step = model_objects['train_step']
    # loss_fn_jit = model_objects['loss_fn_jit']
    loss_fn = model_objects['loss_fn']
    predict = model_objects['predict']
    state = model_objects['state']
    # key = model_objects['key']

    # unpack cfg objects
    batch_size = cfg['batch_size']
    num_labels = cfg['num_labels']
    threads = cfg['threads']
    # fraction = cfg['fraction']
    jit = cfg['jit']

    if jit:
        with Spinner('Warming up JIT...'):
            warmup_images = jnp.ones((batch_size, 224, 224, 3), dtype=jnp.float32)
            warmup_labels = jnp.ones((batch_size, num_labels), dtype=bool)
            # no need to log the cost here

            jitted = jax.jit(loss_fn)
            lowered = jitted.lower(state.params, warmup_images, warmup_labels)
            compiled = lowered.compile()
            loss_fn_jit = compiled


            jitted = jax.jit(predict)
            lowered = jitted.lower(state.params, warmup_images)
            compiled = lowered.compile()
            predict_jit = compiled

            del warmup_images
            del warmup_labels
            del jitted
            del lowered
            del compiled  # references to jitted functions should still remain
    else:
        loss_fn_jit = loss_fn
        predict_jit = predict

    print('Loading best checkpoint')
    logging.info('Loading best checkpoint')
    best_state = checkpointer.restore(state)

    mimic_loader = make_mimic_dataloader(
        threads=threads,
        batch_size=batch_size,
        cache=cache,
        labels=num_labels,
        # frac=0.001,
    )

    print('External test phase start')
    print('Test loss')
    logging.info('Start test loop')
    test_time = timer()
    test_loss = test_loop(best_state, mimic_loader, loss_fn, loss_fn_jit)
    test_time = timer() - test_time
    logging.info('End test loop')


    print('Get test predictions')
    logging.info('Start predict')
    pred_time = timer()
    preds = predict_loop(best_state, mimic_loader, predict, predict_jit)
    pred_time = timer() - pred_time
    logging.info('End predict')

    logging.info('Save test results')
    # only three items here, open directly
    with open(cfg['mimic_results_dir'] / 'mimic_test_loss_time.txt', 'w') as f:
        f.write(f'test_loss\t{test_loss}\n')
        f.write(f'test_time\t{test_time}\n')
        f.write(f'pred_time\t{pred_time}\n')

    logging.info('Save test predictions')
    preds_df = pd.DataFrame(preds, index=mimic_loader.image_paths, columns=mimic_loader.label_cols)
    preds_df.index.name = 'path'
    preds_file = cfg['mimic_results_dir'] / 'mimic_preds.csv'
    preds_df.to_csv(preds_file)

    print('Test phase end')
    print(f'Results: {cfg["mimic_results_dir"]}')

if __name__ == '__main__':
    from safetensor_diskcache import make_cache
    cache_dir = '/srv/store/Projects/schan/quantum/.cache/'
    cache = make_cache(cache_dir)
    from tqdm import tqdm
    # cache
    mimic_loader = make_mimic_dataloader(
        threads=1,
        batch_size=4,
        cache=cache,
        labels=19,
    )
    for _ in tqdm(mimic_loader):
        pass
