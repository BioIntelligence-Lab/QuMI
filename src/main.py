import logging
import pathlib
from timeit import default_timer as timer

import click
import cloup
import config
import utils
from cloup import option, option_group

# TODO: instead of using default choices, use a default dictionary, then if an option is not passed, the default value will be pulled from the dict instead


CONFIG = '/srv/store/Projects/schan/quantum/config.yml'
logger = logging.getLogger(__name__)  # logging is blocking, but we don't log inside of a loop

experiment_options = option_group(
    'Experiment parameters',
    option(
        '-c',
        '--classifier',
        default='classical',
        type=click.Choice(['classical', 'quantum']),
        show_default=True,
        help='Classification head',
    ),
    option(
        '-n',
        '--num-labels',
        default=8,
        type=click.IntRange(1, 19),
        show_default=True,
        help='Number of labels to classify',
    ),
    option(
        '-f',
        '--fraction',
        default=0.01,
        type=click.FloatRange(0, 1),
        show_default=True,
        help='The fraction of data to use',
    ),
    option(
        '--freeze/--unfreeze',
        default=False,
        show_default=True,
        help='Whether to freeze the weights of the transferred model',
    ),
)

hyperparameter_options = option_group(
    'Hyperparameters',
    option('--seed', default=42, type=int, show_default=True),
    option('--learning_rate', default=1e-4, type=float, show_default=True),
    option(
        '-l',
        '--num-layers',
        default=3,
        type=int,
        show_default=True,
        help='Number of layers for quantum model (always forced to 0 for classical)',
    ),
)

training_options = option_group(
    'Training options',
    option(
        '--epochs',
        default=50,
        type=int,
        show_default=True,
        help='Maximum number of epochs to train for',
    ),
    option(
        '--patience',
        default=5,
        type=int,
        show_default=True,
        help='Number of epochs w/o improvement before early stopping',
    ),
)

performance_options = option_group(
    'Performance options',
    option('-b', '--batch-size', default=32, type=int, show_default=True),
    option(
        '-g',
        '--gpu-num',
        default='best',
        type=str,
        show_default=True,
        help='The GPU to use (set to None for CPU, or best to auto-select a free one)',
    ),
    option('--jit/--no-jit', default=True, show_default=True, help='Whether to use JIT'),
    option(
        '-t',
        '--threads',
        default=0,
        type=int,
        show_default=True,
        help='Number of threads to use (set to 0 to use batch size / 4)',
    ),
)


directory_options = option_group(
    'Directory options',
    option(
        '--cache-dir',
        type=click.Path(file_okay=False, path_type=pathlib.Path),
        default='/srv/store/Projects/schan/quantum/.cache/',
        show_default=True,
        help='cache directory to use',
    ),
    option(
        '--clobber/--no-clobber',
        default=False,
        show_default=True,
        help='Whether to overwrite experiment if it exists already',
    ),
    option(
        '--experiment-dir',
        type=click.Path(file_okay=False, path_type=pathlib.Path),
        default='/srv/store/Projects/schan/quantum/experiments/ISVLSI/',
        show_default=True,
        help='directory to store results in',
    ),
)


# Use shared options: https://stackoverflow.com/questions/40182157/shared-options-and-flags-between-commands
def add_options(options):
    def _add_options(func):
        for opt in reversed(options):
            func = opt(func)
        return func

    return _add_options


shared_options = add_options(
    [experiment_options, hyperparameter_options, performance_options, directory_options],
)


CONTEXT_SETTINGS = {'help_option_names': ['-h', '--help']}


@cloup.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """Script for ML training and testing"""


def setup_directories_from_config(**kwargs):
    cfg = config.get_config(CONFIG, kwargs)
    if cfg['classifier'] == 'classical':
        cfg['num_layers'] = 0

    params_str = utils.parameter_str(cfg)
    hparams_str = utils.hyperparameter_string(cfg)
    import pprint

    pp = pprint.PrettyPrinter()
    # pp.pprint(cfg)

    utils.setup_jax_devices(cfg['gpu_num'])

    checkpoint_dir = utils.mkdir_p(
        cfg['experiment_dir'], params_str, hparams_str, 'checkpoints', clobber=cfg['clobber']
    )
    results_dir = utils.mkdir_p(
        cfg['experiment_dir'],
        params_str,
        hparams_str,
        'results',
        clobber=cfg['clobber'],
    )
    log_dir = utils.mkdir_p(
        cfg['experiment_dir'],
        params_str,
        hparams_str,
        'logs',
        clobber=cfg['clobber'],
    )

    cost_dir = utils.mkdir_p(
        cfg['experiment_dir'],
        params_str,
        hparams_str,
        'cost',
        clobber=cfg['clobber'],
    )
    cache_dir = utils.mkdir_p(
        cfg['cache_dir'],
        clobber=True,  # always ok to "clobber" cache
    )

    cfg['checkpoint_dir'] = checkpoint_dir
    cfg['results_dir'] = results_dir
    cfg['log_dir'] = log_dir
    cfg['cost_dir'] = cost_dir
    cfg['cache_dir'] = cache_dir

    pp.pprint(cfg)

    from checkpointer import Checkpointer
    from safetensor_diskcache import make_cache

    cache = make_cache(cache_dir)
    checkpointer = Checkpointer(checkpoint_dir)

    logging.basicConfig(filename=log_dir / 'log.txt', filemode='w', level=logging.INFO)

    import jax.random as jrand
    from model import make_model
    from vendored.spinner import Spinner

    key = jrand.PRNGKey(cfg['seed'])
    key, model_key = jrand.split(key, num=2)

    with Spinner('Loading model'):
        model, train_step, loss_fn, predict, state = make_model(
            key,
            num_labels=cfg['num_labels'],
            num_layers=cfg['num_layers'],
            head=cfg['classifier'],
            learning_rate=cfg['learning_rate'],
            freeze=cfg['freeze'],
        )

    # TODO: convert to Dataclass
    model_objects = {
        'cache': cache,
        'checkpointer': checkpointer,
        'model': model,
        'train_step': train_step,
        'loss_fn': loss_fn,
        'predict': predict,
        'state': state,
        'key': key,
    }
    return cfg, model_objects



@cli.command('train', short_help='train model')
@training_options
@shared_options
def train(**kwargs):
    """Train the model. Directory path names will be inferred from the hyper/parameters"""
    total_time = timer()
    cfg, model_objects = setup_directories_from_config(**kwargs)
    # save hparams

    import json
    with open(cfg['results_dir'] / 'params.json', 'w') as fp:
        json.dump(sorted(cfg), fp)

    import trainer
    model_objects = trainer.train(cfg, model_objects)
    trainer.test(cfg, model_objects)
    # save jax allocations

    logging.info('Saving memory profile!')
    import jax
    jax.profiler.save_device_memory_profile(cfg['results_dir'] / 'memory.pprof')
    total_time = timer() - total_time
    with open(cfg['results_dir'] / 'total_time.txt', 'w') as fp:
        fp.write(f'total_time\n{total_time}\n')
    # now we are really done
    logging.info('Done!')


@cli.command('test', short_help='test model')
@training_options
@shared_options
def train(**kwargs):
    """Test the model on NIH. Directory path names will be inferred from the hyper/parameters"""
    total_time = timer()
    cfg, model_objects = setup_directories_from_config(**kwargs)
    # save hparams

    import json
    with open(cfg['results_dir'] / 'params.json', 'w') as fp:
        json.dump(sorted(cfg), fp)

    import trainer
    trainer.test(cfg, model_objects)

    # now we are really done
    logging.info('Done!')


@cli.command('testmimic', short_help='test model checkpoint with mimic')
@shared_options
def test(**kwargs):
    """Test the model. Directory path names will be inferred from the hyper/parameters"""
    total_time = timer()
    cfg, model_objects = setup_directories_from_config(**kwargs)

    params_str = utils.parameter_str(cfg)
    hparams_str = utils.hyperparameter_string(cfg)
    # add mimic results
    mimic_results_dir = utils.mkdir_p(
        cfg['experiment_dir'],
        'mimic',
        params_str,
        hparams_str,
        clobber=cfg['clobber'],
    )
    cfg['mimic_results_dir'] = mimic_results_dir
    # should override the prior log
    logging.basicConfig(filename = cfg['mimic_results_dir'] / 'log.txt', filemode='w', level=logging.INFO)

    # save hparams
    import json
    with open(cfg['mimic_results_dir'] / 'params.json', 'w') as fp:
        json.dump(sorted((str(k), str(v)) for k, v in cfg.items()), fp)

    import mimic
    mimic.external_test(cfg, model_objects)

    with open(cfg['mimic_results_dir'] / 'total_time.txt', 'w') as fp:
        fp.write(f'total_time\n{total_time}\n')
    # now we are really done
    logging.info('Done!')



@cli.command('cache', short_help='build the cache')
@cloup.argument(
    'stage',
    default='all',
    type=click.Choice(['all', 'train', 'val', 'test']),
    help='which parts of the cache to build',
)
@cloup.confirmation_option(prompt='Are you sure you want to build the cache?')
@directory_options
def cache(**kwargs):
    cfg = config.get_config(CONFIG, kwargs)
    cache_dir = cfg['cache_dir']
    stage = cfg['stage']
    print(f'Caching with {cache_dir}')
    utils.build_cache(cache_dir, stage)


@cli.command('draw', short_help='draw model')
@shared_options
def draw(**kwargs):
    """Draw the model"""
    import jax
    cfg, model_objects = setup_directories_from_config(**kwargs)
    model = model_objects['model']
    print(model.tabulate(jax.random.key(0), jax.numpy.empty((cfg['batch_size'], 224, 224, 3)), compute_flops=True, compute_vjp_flops=True, depth=1))

if __name__ == '__main__':
    cli()
