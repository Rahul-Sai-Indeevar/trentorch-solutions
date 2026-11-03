import numpy as np


def temperature_calibrator(z: np.ndarray, tau: float) -> np.ndarray:
    """
    z: logits.
    tau: temperature, tau > 0.

    p = softmax(z / tau), using an overflow-safe softmax (subtract the
    max before exponentiating). Divide by tau, never multiply.

    Return p.
    """
    # TODO: scale by dividing, then stable softmax.
    z_scaled = z / tau
    exp_z = np.exp(z_scaled - np.max(z_scaled, axis=-1, keepdims=True))
    p = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    return p
