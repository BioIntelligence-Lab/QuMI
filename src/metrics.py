import chex
import optax
from jax import numpy as jnp

# alternatively, compute the inverse sigmoid then pass into sigmoid_binary_cross_entropy


def binary_cross_entropy_with_inverse(probs, labels):
    # https://stackoverflow.com/questions/10097891/inverse-logistic-sigmoid-function
    logits = jnp.log(probs) - jnp.log(1 - probs)
    return optax.sigmoid_binary_cross_entropy(logits, labels)


def binary_cross_entropy(probs, labels):
    """Computes sum of element-wise cross entropy given probs and labels.

    A variation of https://github.com/google-deepmind/optax/blob/main/optax/losses/_classification.py#L25#L60 where probabilities should be applied beforehand.

    Args:
      logits: Each element is the unnormalized log probability of a binary
        prediction. See note about compatibility with `labels` above.
      labels: Binary labels whose values are {0,1} or multi-class target
        probabilities. See note about compatibility with `logits` above.

    Returns:
      cross entropy for each binary prediction, same shape as `logits`.
    """
    chex.assert_type([probs], float)
    labels = labels.astype(probs.dtype)
    # sigmoid clamped between 1 and 0, which could have log(0). Clamp values by dtype epsilon
    eps = jnp.finfo(probs.dtype).eps
    probs = jnp.clip(probs, eps, 1.0 - eps)
    log_p = jnp.log(probs)
    log_not_p = jnp.log(1.0 - probs)
    return -labels * log_p - (1.0 - labels) * log_not_p


if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import log_loss

    # binary class
    sigmoids = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 1, 0, 1])
    # binary solved by np.mean
    jans = jnp.sum(binary_cross_entropy(sigmoids, labels))
    skans = log_loss(labels, sigmoids, normalize=False)
    if not np.allclose(jans, skans):
        print(jans, skans)
    # multilabel doesn't work properly for sklearn, sum the zips
    sigmoids = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    labels = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
    # verify(sigmoids, labels)
    jans = jnp.sum(binary_cross_entropy(sigmoids, labels))
    skans = sum(log_loss(i, j, normalize=False) for i, j in zip(labels, sigmoids))
    if not np.allclose(jans, skans):
        print(jans, skans)
    print(jnp.sum(binary_cross_entropy_with_inverse(sigmoids, labels)))
    print(jnp.sum(jans))
