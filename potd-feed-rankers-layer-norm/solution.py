import numpy as np


def feed_ranker_layer_norm(x: np.ndarray, eps: float) -> np.ndarray:
    """
    x: length-d feature vector.
    eps: small positive constant.

    mu = mean(x)
    v = population variance of x (divide by d, not d - 1)
    xhat_i = (x_i - mu) / sqrt(v + eps)

    No learnable scale or bias. Return xhat.
    """
    # TODO: population variance, eps added inside the sqrt.
    mu = np.mean(x)
    v = np.var(x)
    return (x - mu) / np.sqrt(v + eps)
