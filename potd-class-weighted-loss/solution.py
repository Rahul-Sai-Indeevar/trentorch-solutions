import numpy as np


def class_weighted_loss(
    logits: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> float:
    """
    logits: n x C logits.
    labels: length-n 0-based true class indices.
    weights: length-C per-class weights.

    p_ic = softmax(logits_i)_c
    L = -(1/n) * sum_i weights[labels[i]] * log(p_i, labels[i])

    Divide by n, NOT by sum(weights[labels]). Return L.
    """
    # TODO: stable log-softmax per row, weight by true-class weight,
    # sum, divide by n.
    
    n = len(labels)
    C = len(weights)

    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    log_sum_exp = np.log(np.sum(exp_logits, axis=1, keepdims=True))
    log_softmax = (logits - logits_max) - log_sum_exp

    true_log_probs = log_softmax[np.arange(n), labels]
    sample_weights = weights[labels]

    weighted_loss = sample_weights * true_log_probs
    L = -np.sum(weighted_loss) / n

    return L
