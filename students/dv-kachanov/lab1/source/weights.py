import numpy as np


def zeros_init(n_features: int) -> np.ndarray:
    return np.zeros(n_features, dtype=float)


def random_init(
    n_features: int,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    bound = 1.0 / (2.0 * n_features)

    return rng.uniform(
        -bound,
        bound,
        size=n_features
    )


def correlation_init(
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    numerator = X.T @ y
    denominator = np.sum(X ** 2, axis=0)

    return numerator / (denominator + eps)
