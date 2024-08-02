from diskcache import UNKNOWN, Cache, Disk
from safetensors.flax import load, save


def make_safetensors_disk(compress='zstd', compress_level=1):
    """Return a new Disk class for caching with safetensors.

    Safetensors is a binary format which can save space compared to PNGs.
    Zstd compression saves even more space, and de/compress speed is faster than I/O.
    """
    # work around disk_* parameters being forgetten
    # https://github.com/grantjenks/python-diskcache/issues/309

    match compress:
        case 'zlib':
            import zlib

            lib = zlib
        case 'zstd':
            import pyzstd

            lib = pyzstd
        case 'lz4':
            import lz4.frame

            lib = lz4.frame
        case _:
            lib = None

    # create a new function for compress/decompress
    if lib is None:
        compress_fn = lambda x: x
        decompress_fn = lambda x: x
    else:
        compress_fn = lambda x: lib.compress(x, compress_level)
        decompress_fn = lib.decompress

    class SafeTensorsDisk(Disk):
        """Cache key and value using HuggingFace safetensors."""

        def __init__(self, directory=None, **kwargs):
            """Initialize SafeTensors disk instance.

            :param kwargs: super class arguments
            """
            super().__init__(directory, **kwargs)

        def store(self, value, read, key=UNKNOWN):
            if not read:
                st_bytes = save(
                    {'': value}
                )  # save requires a dictionary, but we only serialize 1 value
                value = compress_fn(st_bytes)
            return super().store(value, read, key=key)

        def fetch(self, mode, filename, value, read):
            data = super().fetch(mode, filename, value, read)
            if not read:
                st_bytes = decompress_fn(data)
                data = load(st_bytes)['']
            return data

    return SafeTensorsDisk


def make_cache(cache_dir):
    """Helper function to make an 'unlimited' cache without culling. 1 TB of cache should be plenty."""
    return Cache(directory=cache_dir, disk=make_safetensors_disk(), cull_limit=0, size_limit=2**40)


if __name__ == '__main__':
    import jax.numpy as jnp
    from diskcache import Cache

    a = jnp.array(1)
    for compress in [None, 'zlib', 'zstd', 'lz4']:
        cache = Cache(disk=make_safetensors_disk(compress=compress), cull_limit=0)
        cache['1'] = a
        print(cache['1'])
