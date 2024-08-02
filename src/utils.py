def benchmark(f, args=None, rounds=100, warmup = 10):
    from timeit import default_timer as timer  # always perf_counter
    import gc

    # from tqdm import trange
    print(f'Benchmarking {f.__name__}, {args}')

    if args is None:
        args = []

    for i in range(warmup): # warmup
        f(*args)

    times = []
    gc.disable()  # avoid GC overhead
    gc.collect() # ensure enough memory
    for i in range(rounds):
        start = timer()
        f(*args)
        end = timer()
        times.append(end - start)
    gc.enable()
    return times


# https://stackoverflow.com/a/59571639
def get_gpu_memory():
    import subprocess as sp

    command = 'nvidia-smi --query-gpu=memory.free --format=csv'
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    return [
        int(x.split()[0]) for i, x in enumerate(memory_free_info)
    ]


# https://stackoverflow.com/a/6422754
def list_gpus_by_free_memory():
    return [i[0] for i in sorted(enumerate(get_gpu_memory()), key=lambda x: x[1])][::-1]


def get_most_free_gpu():
    gpu_mem = get_gpu_memory()
    return gpu_mem.index(max(gpu_mem))


def tree_equals(tree1, tree2):
    import jax.tree as jtree
    import tree_math as tm

    v1 = tm.Vector(tree1)
    v2 = tm.Vector(tree2)
    return all(arr.all() for arr in jtree.flatten(v1 == v2)[0])


# TODO: does this belong in a different module?
def build_cache(cache_dir, stage):
    from dataloader import make_dataloader
    from safetensor_diskcache import make_cache
    from tqdm import tqdm

    cache = make_cache(cache_dir)

    def noop(loader):
        for _ in tqdm(loader):
            pass

    # batch size 4 and 1 thread is the fastest. probably because of multiple concurrent writes slowing the cache down.
    if stage in ('all', 'val'):
        print('Caching val loader')
        val_loader = make_dataloader('val', cache=cache)
        noop(val_loader)

    if stage in ('all', 'test'):
        print('Caching test loader')
        test_loader = make_dataloader('test', cache=cache)
        noop(test_loader)

    if stage in ('all', 'train'):
        print('Caching train loader')
        train_loader = make_dataloader('train', cache=cache)
        noop(train_loader)


def setup_jax_devices(gpu_num='best', use_64=False):
    """initializes env vars for controlling jax"""
    import os

    # if on a shared HPC system, don't hog memory. Otherwise, enable it to prevent GPU memory fragmenting which can slow things down. BUT! if you use more than this amount of memory, you might OOM, bc it might not allocate you more memory than this
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # for minimal GPU usage, enable deallocations at the cost of performance
    # os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    # XLA performance flags don't seem to do anything on a single GPU
    # https://jax.readthedocs.io/en/latest/gpu_performance_tips.html
    # https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/rosetta/projects/pax/README.md#xla-flags

    if gpu_num == 'best':
        gpu_num = str(get_most_free_gpu())
        print(f'Using gpu with most free memory: {gpu_num}')
    elif not gpu_num.isdigit():  # cpu only
        gpu_num = ''

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

    # set gpu determinism with performance penalty
    # Deterministic scatter is not implemented yet. Results may not be reproducible across multiple runs.
    # https://github.com/google/jax/pull/4824/files#r519817635
    # https://github.com/google/jax/discussions/10674
    # os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

    import jax

    print(f'Using devices: {jax.devices()}')

    if use_64:
        jax.config.update(
            'jax_enable_x64', True
        )  # some pennylane ops may require this, but it inflates the compile/runtime for large quantum circuits


def mkdir_p(*tree, clobber=False):
    """creates directory with this tree, and returns the path as a Path object"""
    from pathlib import Path

    path = Path(*(str(t) for t in tree))
    if path.exists():
        print(f'{path} exists and clobber={clobber}')
        if not clobber:
            import sys

            sys.exit()
    else:
        print(f'Creating {path}')
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def param_str(config: dict, params: list):
    """given a dictionary, picks out the params and returns a formatted string."""
    these_params = {k: config[k] for k in params}
    return '-'.join('{}={}'.format(k, v) for k, v in sorted(these_params.items()))


def parameter_str(cfg: dict):
    return param_str(cfg, cfg['parameters'])


def hyperparameter_string(cfg: dict):
    return param_str(cfg, cfg['hyperparameters'])


# TODO: move these to a separate jax utils


def pad_batch_to_size(batch, batch_size):
    import jax.numpy as jnp

    # pad with 0s, without using pad, shard, unpad: https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#pad_shard_unpad
    return jnp.concatenate((batch, jnp.zeros((batch_size - batch.shape[0], *batch.shape[1:]))))


# because there's no way to set dtypes as global...
def make_jnp_array(dtype):  # noqa: ARG001
    import jax.numpy as jnp

    def _make_jnp_array(dtype=jnp.float32, *args, **kwargs):
        return jnp.array(*args, **kwargs, dtype=dtype)

    return _make_jnp_array


# https://jax.readthedocs.io/en/latest/debugging/print_breakpoint.html#usage-with-jax-lax-cond
def breakpoint_if_nonfinite(x):
    is_finite = jnp.isfinite(x).all()

    def true_fn(x):
        pass

    def false_fn(x):
        jax.debug.breakpoint()

    jax.lax.cond(is_finite, true_fn, false_fn, x)


def tree_equals(tree1, tree2):
    import tree_math as tm

    v1 = tm.Vector(tree1)
    v2 = tm.Vector(tree2)
    return all([arr.all() for arr in jax.tree.flatten(v1 == v2)[0]])
