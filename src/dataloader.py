from itertools import batched
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
from diskcache import Cache
from image_processor import load_image_hf, random_augment_image

# based off of https://github.com/BirkhoffG/jax-dataloader
from joblib import Parallel, delayed
from nihcxr import dataset, get_labels
from safetensor_diskcache import make_safetensors_disk

import numpy as np
from vendored.spinner import Spinner

# as long as the dataloader is emptied, memory usage is constant
# this might not happen if compilation time takes too long, as images will overflow the GPU memory
def epoch_iterator(
    image_paths: Sequence[str],
    labels: Sequence,
    batch_size: int,
    parallel,
    indices: Sequence[int],
    load_fn: Callable,
    augment_key=None,
):

    def key_gen(key, num):
        curr_key = key
        for i in range(num):
            curr_key, next_key = jrand.split(curr_key)
            yield next_key
    
    these_paths = [image_paths[i] for i in indices]
    these_labels = [labels[i] for i in indices]

    if augment_key is None:
        these_images = parallel(delayed(load_fn)(path) for path in these_paths)

        def get_images(paths):
            return parallel(delayed(load_fn)(path) for path in paths)
    else:
        def key_gen(key, num):
            curr_key = key
            for i in range(num):
                curr_key, next_key = jrand.split(curr_key)
                yield next_key

        # with keygen to lower memory
        augment_keys = key_gen(augment_key, len(indices))
        these_images = parallel(delayed(load_fn)(next(augment_keys), path) for path in these_paths)
        # or generate keys aot
        # augment_keys = jrand.split(augment_key, num=len(indices))
        # these_images = parallel(delayed(load_fn)(key, path) for key, path in zip(augment_keys, these_paths, strict=True))

        def get_images(paths):
            return parallel(delayed(load_fn)(next(augment_keys), path) for path in paths)

    batches = zip(
        # batched(these_paths, batch_size),
        batched(these_images, batch_size),
        batched(these_labels, batch_size),
        strict=True,
    )
    for pixels, lbls in batches:
        yield jnp.array(pixels), jnp.array(lbls)
    return


# TODO: set cache dir to local directory
class DataLoaderJax:
    # modified to read images
    def __init__(
        self,
        image_paths: Sequence[str],  # paths to images
        labels: Sequence,  # labels of images
        label_cols: Sequence[str],  # column names
        batch_size: int = 1,  # batch size
        shuffle: bool = False,  # if true, dataloader shuffles before sampling each batch
        rng: int = 42,  # use this as the initial seed, or pass an rng key
        # drop_last: bool = False,  # if true, drop last batch
        threads: int = 0,
        load_fn: Callable = load_image_hf,
        augment: bool = False,
        cache=None,
    ):
        assert len(image_paths) == len(labels)
        print('Loading dataloader')
        self.label_cols = label_cols
        # convert to proper sequences
        self.image_paths = image_paths.tolist()
        self.labels = jnp.array(labels)
        self.key = rng  #  jrand.key(rng) if isinstance(rng, int) else rng

        self.indices = jnp.array(np.arange(len(labels)))  # should be ints
        self.batch_size = batch_size
        self.shuffle = shuffle
        # self.drop_last = drop_last

        # memoize load_fn if cache available
        this_load_fn = cache.memoize()(load_fn) if cache is not None else load_fn
        # add augmentation after loading from cache
        self.augment = augment
        if self.augment:

            random_augment_image_jit = jax.jit(random_augment_image)
            # warmup the image augmentation. Try blocking before loading all the images with the dataloader
            with Spinner('Warming up image augmentation...'):  # jit barely takes time to warmup
                warmup_images = jnp.ones(
                    (224, 224, 3), dtype=jnp.float32
                )
                warmup_key = jrand.PRNGKey(0)
                random_augment_image_jit(warmup_key, warmup_images)
                del warmup_key
                del warmup_images
                
            self.block = 0
            def load_and_augment(key, path):
                # don't return until the image is ready. otherwise multithreads will load images too quickly and OOM the GPU
                # https://jax.readthedocs.io/en/latest/async_dispatch.html
                return random_augment_image_jit(key, this_load_fn(path)).block_until_ready()

            self.load_fn = load_and_augment

        else:
            self.load_fn = this_load_fn

        # jax is incompatible with fork: https://github.com/google/jax/issues/1805
        # this can be bypassed by using threading backend with joblib
        # this backend is suitable for expensive library calls (Jax, Pillow)
        # doesn't cause recompilation of jax functions and can scale up to 16 threads
        self.threads = max(1, batch_size // 4) if threads <= 0 else threads
        self.parallel = Parallel(n_jobs=self.threads, prefer='threads', return_as='generator')
        n_batches, n_rest = divmod(len(self.indices), self.batch_size)
        print(
            f'Loaded dataloader of {n_batches} batches + {n_rest} samples = {len(self.indices)} total, using {self.threads} threads'
        )

    def __iter__(self):
        # shuffle (permutation) indices every epoch
        indices = (
            jrand.permutation(self.next_key(), self.indices).__array__()
            if self.shuffle
            else self.indices
        )
        #if self.drop_last:
        #    indices = indices[: len(self.indices) - len(self.indices) % self.batch_size]

        augment_key = self.next_key() if self.augment else None
        return epoch_iterator(
            self.image_paths,
            self.labels,
            self.batch_size,
            self.parallel,
            indices,
            self.load_fn,
            augment_key,
        )

    def next_key(self):
        self.key, subkey = jrand.split(self.key)
        return subkey

    def __len__(self):
        n_batches, leftovers = divmod(len(self.indices), self.batch_size)
        return n_batches + (leftovers > 0)
        # WRONG! VV
        # return len(self.indices) // self.batch_size + int(not self.drop_last)

    def get_len_items(self):
        return len(self.indices)


def make_dataloader(
    stage,
    key=42,
    # n_samples=0,
    batch_size=4,
    cache=None,
    labels=19,
    threads=0,
    shuffle=False,
    augment=False,
    frac=1,
):
    if key is not None:
        key = jrand.key(key) if isinstance(key, int) else key

    dataloader_key, dataset_key = jrand.split(key)

    label_cols = get_labels(labels) if isinstance(labels, int) else labels

    dataloader_df = dataset.get_df(stage, frac=frac, key=dataset_key)
    paths = dataloader_df.index
    labels = dataloader_df[label_cols]

    return DataLoaderJax(
        image_paths=paths,
        labels=labels,
        batch_size=batch_size,
        shuffle=shuffle,
        augment=augment,
        rng=dataloader_key,
        cache=cache,
        label_cols=label_cols,
        threads=threads,
    )


def test_benchmark():
    # TODO: rework this one
    # prepare cache. cull_limit is 0, otherwise it culls every 10 inserts
    # os.system(f'find {directory} -type f -user $USER -name \'*.val\' 2>/dev/null | xargs du -hc | tail -n1')

    build_cache_times = {}
    reuse_cache_times = {}
    cache_sizes = {}

    batch_size = 64
    threads = 0
    # one run without caching
    dl = DataLoaderJax(
        image_paths=image_paths,
        labels=image_labels,
        batch_size=batch_size,
        shuffle=False,
        rng=0,
        threads=threads,
        cache=None,
    )
    start = timer()
    for pixels, labels in tqdm(dl):
        pass
    end = timer()
    build_cache_times['nocache'] = end - start
    reuse_cache_times['nocache'] = end - start
    cache_sizes['nocache'] = 0

    for compress in [None, 'zlib', 'zstd', 'lz4']:
        # supposed to set disk settings with disk_*, but it doesn't (has never?) worked
        cache = Cache(disk=make_safetensors_disk(compress), cull_limit=0)
        # cache = Cache(disk=SafeTensorsDisk, disk_compress=compress, cull_limit=0)
        directory = cache.directory
        print(f'Initialized cache at {directory}')
        dl = DataLoaderJax(
            image_paths=image_paths,
            labels=image_labels,
            batch_size=batch_size,
            shuffle=False,
            rng=0,
            threads=threads,
            cache=cache,
        )
        start = timer()
        for pixels, labels in tqdm(dl):
            pass
        end = timer()
        build_cache_times[compress] = end - start
        cache_sizes[compress] = cache.volume()
        start = timer()
        for pixels, labels in tqdm(dl):
            pass
        end = timer()
        reuse_cache_times[compress] = end - start
        # print(f'{compress}: reusing cache: {end - start} seconds')
    import pandas as pd

    # reuse_cache_time is most important bc epochs
    print(
        pd.DataFrame(
            {
                'build_cache_time (s)': build_cache_times,
                'reuse_cache_time (s)': reuse_cache_times,
                'cache_size': cache_sizes,
            },
        ).sort_values('reuse_cache_time (s)'),
    )


def test_equality():
    # get first batch
    dl = DataLoaderJax(
        image_paths=image_paths,
        labels=image_labels,
        batch_size=2,
        shuffle=False,
        rng=0,
        threads=0,
    )
    for pixels, labels in tqdm(dl):
        print(pixels.shape, labels.shape)
        p1, l1 = pixels, labels
        break

    for compress in [None, 'zlib', 'zstd', 'lz4']:
        cache = Cache(
            disk=make_safetensors_disk(compress),
            cull_limit=0,
            size_limit=2**40,
        )  # 1TB of disk!! The cache will never be emptied, practically speaking.
        dl = DataLoaderJax(
            image_paths=image_paths,
            labels=image_labels,
            batch_size=2,
            shuffle=False,
            rng=0,
            threads=0,
            cache=cache,
        )

        def check_first_batch_equals(dl):
            for pixels, labels in tqdm(dl):  # add only the first batch
                assert jnp.allclose(pixels, p1) and jnp.allclose(p1, pixels)
                assert jnp.allclose(labels, l1) and jnp.allclose(l1, labels)
                break

        check_first_batch_equals(dl)  # uncached
        check_first_batch_equals(dl)  # cached
    print('Passed correctness test')


def test_augment():
    dl = DataLoaderJax(
        image_paths=image_paths,
        labels=image_labels,
        batch_size=2,
        shuffle=False,
        rng=0,
        threads=0,
        augment=True,
    )
    for pixels, labels in tqdm(dl):
        print(pixels.shape, labels.shape)
        _p1, _l1 = pixels, labels
        break


if __name__ == '__main__':
    from timeit import default_timer as timer

    import nihcxr
    from tqdm import tqdm

    image_paths = nihcxr.get_df('train')['path'][:4]  # .to_numpy() # [:640]
    image_labels = nihcxr.get_df('train')[nihcxr.labels][:4]  # .to_numpy() # [:640]
    from diskcache import Cache
    from safetensor_diskcache import make_safetensors_disk  #  SafeTensorsDisk

    # test_equality()
    # test_benchmark()
    test_augment()
