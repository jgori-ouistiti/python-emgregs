# K === lambda
from typing import Any, Union, List, Optional
import warnings

import numpy
import numpy.typing as npt

import scipy.stats as spst
import scipy.optimize as spopt

import pandas

import statsmodels.api as sm
import statsmodels.stats as sms
import statsmodels.formula.api as smf


rng = numpy.random.default_rng(seed=123)


def emg__neg_llh(
    emg_parameters: List[Union[npt.NDArray[Any], float, numpy.floating]],
    x: Union[npt.NDArray[Any], float, numpy.floating],
):

    if x is None:
        raise ValueError(
            "input data to calculate the log likelihood has not been specified"
        )

    mu, sigma, k = emg_parameters

    K = 1 / (sigma * k)
    individual_likelihoods = spst.exponnorm.logpdf(x, K, loc=mu, scale=sigma)
    individual_likelihoods[~numpy.isfinite(individual_likelihoods)] = 0

    return -numpy.sum(individual_likelihoods)


def emg__mle2(
    x: npt.NDArray[Any],
    lower: List[Union[npt.NDArray[Any], float, numpy.floating]] = None,
    upper: List[Union[npt.NDArray[Any], float, numpy.floating]] = None,
    start: List[Union[npt.NDArray[Any], float, numpy.floating]] = None,
    **kwargs,
):

    if lower is None:
        lower = [numpy.min(x), numpy.std(x) / 10, numpy.abs(0.001 / numpy.mean(x))]
    if upper is None:
        upper = [
            numpy.max(x),
            (numpy.max(x) - numpy.min(x)) / 4,
            numpy.abs(100 / numpy.mean(x)),
        ]

    # Method of moments https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
    scaled_skew = (spst.skew(x) / 2) ** (1 / 3)
    if start is None:
        start = [
            numpy.mean(x) - numpy.std(x) * scaled_skew,
            numpy.sqrt(numpy.abs(numpy.std(x) ** 2 * (1 - scaled_skew ** 2))),
            1 / numpy.std(x) * scaled_skew,
        ]

    if not numpy.isfinite(
        start
    ).all():  # more stringent than an isnan test since also discards infinite values
        start = [
            (lower[0] + upper[0]) / 2,
            (lower[1] + upper[1]) / 2,
            numpy.abs(lower[2] + upper[2]) / 2,
        ]

    res = spopt.minimize(
        emg__neg_llh,
        start,
        args=(x,),
        method="L-BFGS-B",
        bounds=spopt.Bounds(lower, upper),
    )

    for n, _x in enumerate(res.x):
        if _x in lower or _x in upper:
            warnings.warn(
                f"Solution {res.x} has component {n} with value {res.x[n]} on the boundary (Lower bound: {lower[n]} / Upper bound: {upper[n]})"
            )
    return res


def beta_log_fn(beta, x, y, mu, sigma, k):
    beta = numpy.atleast_2d(beta).reshape(-1, 1)
    x = numpy.atleast_2d(x)
    y = y.reshape(
        -1,
    )
    return emg__neg_llh([mu, sigma, k], y - (x @ beta).squeeze())


def emg__reg(x, y, beta=None, sigma=None, k=None, maxit=10000, epsilon=0.0001):
    N = y.shape[0]
    if beta is None:
        lm__out = sm.OLS(y, x)
        beta = lm__out.fit().params
    if sigma is None:
        lm__out = smf.ols(
            formula="y~x", data=pandas.DataFrame({"x": x[:, 1], "y": y})
        ).fit()
        anova_mean_sq = sms.anova.anova_lm(lm__out)["mean_sq"]
        rate = numpy.sqrt(1 / anova_mean_sq[1])
        sigma = rng.exponential(scale=1 / rate, size=(1,))

    if k is None:
        lm__out = smf.ols(
            formula="y~x", data=pandas.DataFrame({"x": x[:, 1], "y": y})
        ).fit()
        a = 1 / numpy.sum(lm__out.resid[lm__out.resid > 0])
        k = numpy.abs(rng.normal(loc=a, size=(1,)))

    mu = 0
    beta_list = [beta]
    Q = [0, -beta_log_fn(beta, x, y, mu, sigma, k)]
    iter = 1
    while iter <= maxit and abs((Q[iter] - Q[iter - 1])) >= epsilon:
        iter += 1
        result = spopt.minimize(
            beta_log_fn,
            beta,
            args=(x, y, mu, sigma, k),
            method="Nelder-Mead",
        )
        beta = result.x
        beta_list.append(beta)

        out = emg__mle2(y - (x @ beta.reshape(-1, 1)).squeeze())
        mu = 0
        sigma, k = out.x[1:]

        Q.append(-beta_log_fn(beta, x, y, mu, sigma, k))

    if iter == maxit + 1:
        warnings.warn("WARNING! NOT CONVERGENT!")

    print(f"number of iterations={iter}")
    return {
        "all.loglik": Q,
        "residual": y - (x @ beta.reshape(-1, 1)).squeeze(),
        "loglik": Q[iter],
        "beta": beta,
        "sigma": sigma,
        "ex.rate": k,
        "beta.list": beta_list,
    }


def func_emg__reg(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    intercept: Optional[bool] = False,
    beta: Optional[npt.ArrayLike] = None,
    sigma: Optional[float] = None,
    alpha: Optional[float] = None,
    **kwargs,
):
    if not intercept:
        x = numpy.concatenate((numpy.ones((x.shape[0], 1)), x), axis=1)
    return emg__reg(x, y, beta=beta, sigma=sigma, k=alpha, **kwargs)


if __name__ == "__main__":
    import function_simulation as sim

    N = 500
    data = sim.sim__emg_reg(n=N, sigma=1, alpha=0.1)
    regfit_emg = func_emg__reg(data["X"], data["Y"], maxit=10000)
