import numpy as np


def compute_marginalize_cartesian(samples: np.ndarray):
    """
    Compute mean and convariance of cartesian coordinates and marginalize heading.
    Marginalize heading from the joint gaussian distribution. Assume heading is in the last position.
    https://www.seas.upenn.edu/~cis520/papers/Bishop_2.3.pdf - 2.3.2
    """
    # Compute mean
    mean = np.mean(samples, axis=0)
    # Compute covariance
    centered = samples - mean
    cov = (1 / samples.shape[0]) * np.sum(centered[:, :, np.newaxis] @ centered[:, np.newaxis, :], axis=0)
    # Marginalize heading
    marginal_cov = cov[:2, :2]
    marginal_mean = mean[:2]
    return marginal_mean, marginal_cov
