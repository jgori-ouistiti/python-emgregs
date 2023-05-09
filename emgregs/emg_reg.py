# K === lambda
from typing import Any, Union, List, Optional, Tuple
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


def emg_neg_llh_heterosked(
    emg_parameters: List[Union[npt.NDArray[Any], float, numpy.floating]],
    resid: Union[npt.NDArray[Any], float, numpy.floating],
    x: Union[npt.NDArray[Any], float, numpy.floating],
):

    if resid is None:
        raise ValueError(
            "input data to calculate the log likelihood has not been specified"
        )

    mu, sigma, *expo_scale = emg_parameters

    K = numpy.sum(expo_scale * x, axis=1) / (sigma)
    individual_likelihoods = spst.exponnorm.logpdf(resid, K, loc=mu, scale=sigma)
    individual_likelihoods[~numpy.isfinite(individual_likelihoods)] = 0

    return -numpy.sum(individual_likelihoods)


def emg_neg_llh(
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


def emg_mle2(
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
    ).all():  # more stringent than an isnan test --> also discards infinite values
        start = [
            (lower[0] + upper[0]) / 2,
            (lower[1] + upper[1]) / 2,
            numpy.abs(lower[2] + upper[2]) / 2,
        ]

    res = spopt.minimize(
        emg_neg_llh,
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


def emg_mle2_heterosked(
    resid: npt.NDArray[Any],
    x: npt.NDArray[Any],
    lower: List[Union[npt.NDArray[Any], float, numpy.floating]] = None,
    upper: List[Union[npt.NDArray[Any], float, numpy.floating]] = None,
    start: List[Union[npt.NDArray[Any], float, numpy.floating]] = None,
    **kwargs,
):

    if lower is None:
        lower = [
            numpy.min(resid),
            numpy.std(resid) / 10,
            numpy.abs(0.001 / numpy.mean(resid)),
        ] + [numpy.abs(0.001 / numpy.mean(resid)) for i in range(x.shape[1] - 1)]
    if upper is None:
        upper = [
            numpy.max(resid),
            (numpy.max(resid) - numpy.min(resid)) / 4,
            numpy.abs(100 / numpy.mean(resid)),
        ] + [numpy.abs(100 / numpy.mean(resid)) for i in range(x.shape[1] - 1)]

    # Method of moments https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
    scaled_skew = (spst.skew(resid) / 2) ** (1 / 3)
    if start is None:
        start = [
            numpy.mean(resid) - numpy.std(resid) * scaled_skew,
            numpy.sqrt(numpy.abs(numpy.std(resid) ** 2 * (1 - scaled_skew ** 2))),
            1 / numpy.std(resid) * scaled_skew,
        ] + [0 for i in range(x.shape[1] - 1)]

    if not numpy.isfinite(
        start
    ).all():  # more stringent than an isnan test since also discards infinite values
        start = [
            (lower[0] + upper[0]) / 2,
            (lower[1] + upper[1]) / 2,
            numpy.abs(lower[2] + upper[2]) / 2,
        ] + [0 for i in range(x.shape[1] - 1)]

    res = spopt.minimize(
        emg_neg_llh_heterosked,
        start,
        args=(resid, x),
        method="L-BFGS-B",
        bounds=spopt.Bounds(lower, upper),
    )

    for n, _x in enumerate(res.x):
        if _x in lower or _x in upper:
            warnings.warn(
                f"Solution {res.x} has component {n} with value {res.x[n]} on the boundary (Lower bound: {lower[n]} / Upper bound: {upper[n]})"
            )
    return res


def beta_log_fn_heterosked(beta, x, y, mu, sigma, expo_scale):
    beta = numpy.atleast_2d(beta).reshape(-1, 1)
    x = numpy.atleast_2d(x)
    y = y.reshape(
        -1,
    )
    return emg_neg_llh_heterosked(
        [mu, sigma, *expo_scale], y - (x @ beta).squeeze(), x
    )


def beta_log_fn(beta, x, y, mu, sigma, k):
    beta = numpy.atleast_2d(beta).reshape(-1, 1)
    x = numpy.atleast_2d(x)
    y = y.reshape(
        -1,
    )

    return emg_neg_llh([mu, sigma, k], y - (x @ beta).squeeze())


def _emg_reg(x, y, beta=None, sigma=None, k=None, maxit=10000, epsilon=0.0001):
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

        out = emg_mle2(y - (x @ beta.reshape(-1, 1)).squeeze())
        mu = 0
        sigma, k = out.x[1:]

        Q.append(-beta_log_fn(beta, x, y, mu, sigma, k))

    if iter == maxit + 1:
        warnings.warn("WARNING! NOT CONVERGENT!")

    print(f"number of iterations={iter}")
    # TODO: should return number of observations and parameters as well to compute AIC and BIC easily

    #### Warning: here ex is exponential parameter expressed in rate version (x * ex)

    return {
        "all.loglik": Q,
        "residual": y - (x @ beta.reshape(-1, 1)).squeeze(),
        "loglik": Q[iter],
        "beta": beta,
        "sigma": sigma,
        "ex": k,
        "beta.list": beta_list,
    }


def _emg_reg_heterosked(
    x, y, beta=None, sigma=None, expo_scale=None, maxit=10000, epsilon=0.0001
):
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

    if expo_scale is None:
        lm__out = smf.ols(
            formula="y~x", data=pandas.DataFrame({"x": x[:, 1], "y": y})
        ).fit()
        a = 1 / numpy.sum(lm__out.resid[lm__out.resid > 0])
        k = numpy.abs(rng.normal(loc=a, size=(1,)))
        expo_scale = numpy.zeros((x.shape[1]))
        expo_scale[0] = 1 / k

    mu = 0
    beta_list = [beta]
    Q = [0, -beta_log_fn_heterosked(beta, x, y, mu, sigma, expo_scale)]
    iter = 1
    while iter <= maxit and abs((Q[iter] - Q[iter - 1])) >= epsilon:
        iter += 1
        result = spopt.minimize(
            beta_log_fn_heterosked,
            beta,
            args=(x, y, mu, sigma, expo_scale),
            method="Nelder-Mead",
        )
        beta = result.x
        beta_list.append(beta)

        out = emg_mle2_heterosked(y - (x @ beta.reshape(-1, 1)).squeeze(), x)
        mu = 0
        sigma, *expo_scale = out.x[1:]

        Q.append(-beta_log_fn_heterosked(beta, x, y, mu, sigma, expo_scale))

    if iter == maxit + 1:
        warnings.warn("WARNING! NOT CONVERGENT!")

    print(f"number of iterations={iter}")
    # TODO: should return number of observations and parameters as well to compute AIC and BIC easily

    #### Warning: here ex is exponential parameter expressed in scale version (x/ex)
    return {
        "all.loglik": Q,
        "residual": y - (x @ beta.reshape(-1, 1)).squeeze(),
        "loglik": Q[iter],
        "beta": beta,
        "sigma": sigma,
        "ex": expo_scale,
        "beta.list": beta_list,
    }


def emg_reg_heterosked(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    intercept: Optional[bool] = False,
    beta: Optional[npt.ArrayLike] = None,
    sigma: Optional[float] = None,
    expo_scale: Tuple[float] = None,
    **kwargs,
):
    if not intercept:
        try:
            x = numpy.concatenate((numpy.ones((x.shape[0], 1)), x), axis=1)
        except ValueError:
            x = numpy.concatenate((numpy.ones((x.shape[0], 1)), x.reshape(-1, 1)), axis=1)
    return _emg_reg_heterosked(
        x, y, beta=beta, sigma=sigma, expo_scale=expo_scale, **kwargs
    )


def emg_reg(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    intercept: Optional[bool] = False,
    beta: Optional[npt.ArrayLike] = None,
    sigma: Optional[float] = None,
    alpha: Optional[float] = None,
    **kwargs,
):
    
    x = numpy.array(x)
    y = numpy.array(y)

    if not intercept:
        try:
            x = numpy.concatenate((numpy.ones((x.shape[0], 1)), x), axis=1)
        except ValueError:
            x = numpy.concatenate((numpy.ones((x.shape[0], 1)), x.reshape(-1, 1)), axis=1)

    k_passed = kwargs.pop("k", None)
    k = k_passed if k_passed is not None else alpha

    return _emg_reg(x, y, beta=beta, sigma=sigma, k=k, **kwargs)


if __name__ == "__main__":
    from emgregs import sim_emg_reg, sim_emg_reg_heterosked
    N = 5000
    data1 = sim_emg_reg(
        xmin=1, xmax=7, n=N, beta=(0.3, 0.15), sigma=0.5, alpha=0.01
    )
    data2 = sim_emg_reg_heterosked(
        xmin=1, xmax=7, n=N, beta=(0.3, 0.15), sigma=0.1, expo_scale=(0.1, 0.1)
    )
    result_dict_homosked = emg_reg(data1["X"], data1["Y"], maxit=10000)
    result_dict_heterosked = emg_reg_heterosked(data2["X"], data2["Y"], maxit=10000)
