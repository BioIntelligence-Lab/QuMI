from timeit import timeit

import pytest
from nihcxr import dataset


def test_dataset_speed():
    # second call should be faster
    assert timeit(lambda: dataset.get_df('train')) >= timeit(lambda: dataset.get_df('train'))
    assert timeit(lambda: dataset.get_df('val')) >= timeit(lambda: dataset.get_df('val'))
    assert timeit(lambda: dataset.get_df('test')) >= timeit(lambda: dataset.get_df('test'))
    assert timeit(lambda: dataset.get_df('metadata')) >= timeit(lambda: dataset.get_df('metadata'))


def test_length():
    d1 = dataset.get_df('val')
    d2 = dataset.get_df('val', frac=0.1, key=1)
    assert len(d1) > len(d2)


def test_equality():
    d1 = dataset.get_df('val', frac=0.1, key=1)
    d2 = dataset.get_df('val', frac=0.1, key=1)
    assert d1.equals(d2)


def test_jrand_key():
    from jax.random import PRNGKey

    d1 = dataset.get_df('val', frac=0.5, key=PRNGKey(1))
    d2 = dataset.get_df('val', frac=0.5, key=PRNGKey(1))
    assert d1.equals(d2)
    d1 = dataset.get_df('train', frac=0.5, key=PRNGKey(1))
    d2 = dataset.get_df('train', frac=0.5, key=PRNGKey(1))
    assert d1.equals(d2)
    d1 = dataset.get_df('test', frac=0.5, key=PRNGKey(1))
    d2 = dataset.get_df('test', frac=0.5, key=PRNGKey(1))
    assert d1.equals(d2)


def test_default_arg():
    dataset.get_df('val', frac=0.5)


@pytest.mark.skip(reason='Slow')
def test_data_exists():
    import os.path

    for path in dataset.get_df['metadata'].index:
        assert os.path.isfile(path)  # noqa: PTH113

    print('All images exist!')
