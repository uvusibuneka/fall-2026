import numpy as np


def zeros_init(n_features: int) -> np.ndarray:
    """wj := 0 для всех j = 0, ..., n (вариант 1 из лекции)."""
    return np.zeros(n_features, dtype=float)


def random_init(
    n_features: int,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """Малые случайные значения wj := random(-1/2n, 1/2n) (вариант 2)."""
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
    y: np.ndarray
) -> np.ndarray:
    """Точная дробь wj := <y, fj> / <fj, fj> (вариант 3 из лекции).

    Оптимальна при квадратичной потере и некоррелированных признаках.
    Никакого eps в знаменателе: формула применяется буквально.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    numerator = X.T @ y
    denominator = np.sum(X ** 2, axis=0)

    return numerator / denominator
