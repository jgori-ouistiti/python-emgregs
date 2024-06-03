# Known to work with Python 3.8

from typing import Any, Tuple, Dict
import numpy
import numpy.typing as npt


# Everywhere, lambda is replaced by k (protected word in Python)


# Init RNG
rng = numpy.random.default_rng(seed=123)


def exg_pdf_residuals(
    shape: Tuple[int] = (1,), mu: float = 0, k: float = 1, sigma: float = 1
) -> npt.NDArray[Any]:
    """Generates random emg deviates"""
    return rng.normal(loc=mu, scale=sigma, size=shape) + rng.exponential(
        scale=1 / k, size=shape
    )


def exg_pdf_residuals_heterosked(
    x: npt.NDArray[Any],  # with intercept
    shape: Tuple[int] = (1,),
    mu: float = 0,
    expo_scale: Tuple[float] = (1, 2),
    sigma: float = 1,
) -> npt.NDArray[Any]:
    """Generates random emg deviates"""
    return rng.normal(loc=mu, scale=sigma, size=shape) + rng.exponential(
        scale=numpy.sum((expo_scale * x), axis=1), size=shape
    )


def sim_emg_reg_heterosked(
    xmin: npt.NDArray[Any] = 0.5,
    xmax: npt.NDArray[Any] = 4,
    X: npt.NDArray[Any] = None,
    n: int = 50,
    beta: Tuple[float] = (2, 3, 5),  # Or array
    sigma: float = 0.5,
    expo_scale: Tuple[float] = (1, 2, 0),
) -> Dict[npt.NDArray[Any], npt.NDArray[Any]]:
    """Scale of the exponential is expo_scale.T @ X"""

    expo_scale = numpy.asarray(expo_scale)
    if (expo_scale * xmin <= 0).all() or (expo_scale * xmax <= 0).all():
        raise ValueError(
            f"Input scale should be strictly positive, but equals {expo_scale} ({expo_scale * xmin}; {expo_scale * xmax})"
        )

    if sigma <= 0:
        raise ValueError(f"Input sigma should be strictly positive, but equals {sigma}")

    beta = numpy.asarray(beta).reshape(-1, 1)
    m = beta.shape[0]

    if n is None:
        if X is None:
            n = 50
        else:
            n = len(X)

    if X is None:
        X = numpy.zeros((n, m))
        X[:, 0] = 1
        X[:, 1:] = (xmax - xmin) * rng.random(size=(n, m - 1)) + xmin

    # Add EMG Noise
    E = exg_pdf_residuals_heterosked(X, shape=(n,), sigma=sigma, expo_scale=expo_scale)
    Y = (X @ beta).squeeze() + E
    X = X[:, 1:]

    return {"X": X, "Y": Y}


def sim_emg_reg(
    xmin: npt.NDArray[Any] = 0.5,
    xmax: npt.NDArray[Any] = 4,
    n: int = 50,
    beta: Tuple[float] = (2, 3, 5),  # Or array
    sigma: float = 0.5,
    alpha: float = 0.01,
) -> Dict[npt.NDArray[Any], npt.NDArray[Any]]:

    if alpha <= 0:
        raise ValueError(f"Input alpha should be strictly positive, but equals {alpha}")

    if sigma <= 0:
        raise ValueError(f"Input sigma should be strictly positive, but equals {sigma}")

    beta = numpy.asarray(beta).reshape(-1, 1)
    m = beta.shape[0]

    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.random.html

    if n is None:
        if X is None:
            n = 50
        else:
            n = len(X)

    if X is None:
        X = numpy.zeros((n, m))
        X[:, 0] = 1
        X[:, 1:] = (xmax - xmin) * rng.random(size=(n, m - 1)) + xmin

    # Add EMG Noise
    E = exg_pdf_residuals(shape=(n,), sigma=sigma, k=alpha)
    Y = (X @ beta).squeeze() + E
    X = X[:, 1:]

    return {"X": X, "Y": Y}


def sim_flare_reg(
    xmin: npt.NDArray[Any] = 0.5,
    xmax: npt.NDArray[Any] = 4,
    n: int = 50,
    beta: Tuple[float] = (2, 3, 5),  # Or array
    k: Tuple[float] = (0.5,),
    sigma: float = 0.5,
    alpha: float = 0.01,
) -> Dict[npt.NDArray[Any], npt.NDArray[Any]]:

    if alpha <= 0:
        raise ValueError(f"Input alpha should be strictly positive, but equals {alpha}")

    if sigma <= 0:
        raise ValueError(f"Input sigma should be strictly positive, but equals {sigma}")

    beta = numpy.asarray(beta).reshape(-1, 1)
    m = beta.shape[0]

    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.random.html

    if n is None:
        if X is None:
            n = 50
        else:
            n = len(X)

    if X is None:
        X = numpy.zeros((n, m))
        X[:, 0] = 1
        X[:, 1:] = (xmax - xmin) * rng.random(size=(n, m - 1)) + xmin

    # Add Noise mixture (flare)
    k_zero = k[0]
    condition = rng.random(size=(n,)) < k_zero
    normal = rng.normal(loc=0, scale=sigma, size=(n,))
    expo = rng.exponential(scale=1 / alpha, size=(n,))
    E = numpy.zeros((n,))
    E[condition] = normal[condition]
    E[~condition] = expo[~condition]

    Y = (X @ beta).squeeze() + E
    X = X[:, 1:]

    return {"X": X, "Y": Y}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xmin = -2
    xmax = 4
    n = 5000
    beta = (2, 3, 5)
    sigma = 0.5
    alpha = 0.01
    # If using a vector of input X values, it should match the size of beta. E.g. with beta = (0,1), you should use [1,x] (eg with statsmodels add_constant)
    x, y = sim_emg_reg(
        xmin=xmin, xmax=xmax, n=n, beta=beta, sigma=sigma, alpha=alpha
    ).values()
    fig = plt.figure()
    axx = fig.add_subplot(221)
    axy = fig.add_subplot(222)
    axx.plot(x[:, 0], x[:, 1], "ko")
    axy.plot(range(len(y)), y, "ko")

    x, y = sim_flare_reg(
        xmin=xmin, xmax=xmax, n=n, beta=beta, sigma=sigma, alpha=alpha
    ).values()

    axx = fig.add_subplot(223)
    axy = fig.add_subplot(224)
    axx.plot(x[:, 0], x[:, 1], "ko")
    axy.plot(range(len(y)), y, "ko")
    plt.show()
