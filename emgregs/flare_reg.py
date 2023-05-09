from typing import Any, Dict, Union, List, Optional
import warnings

import numpy
import numpy.typing as npt
import numpy.linalg as npla

import scipy.stats as spst
import scipy.optimize as spopt

import pandas

from sklearn.cluster import KMeans

import statsmodels.api as sm
import statsmodels.stats as sms
import statsmodels.formula.api as smf

from emgregs import emg_reg


rng = numpy.random.default_rng(seed=123)


class mixEM(dict):
    pass


def loglik(res, sigma, k, alpha):
    try:
        k = k[0]
    except IndexError:
        pass
    likelihood_individual_mix = k * spst.norm.pdf(res, scale=numpy.sqrt(sigma)) + (
        1 - k
    ) * spst.expon.pdf(res, scale=1 / alpha)
    return numpy.sum(numpy.log(likelihood_individual_mix))


def Q(res, sigma, k, alpha, z):
    try:
        k = k[0]
    except IndexError:
        pass
    return (
        numpy.sum(z * numpy.log(k))
        + numpy.sum((1 - z) * numpy.log(1 - k))
        - numpy.log(2 * numpy.pi * sigma) * numpy.sum(z) / 2
        - numpy.sum(z * res ** 2) / 2 / sigma
        + numpy.log(alpha) * numpy.sum(1 - z)
        - alpha * numpy.sum((1 - z) * res)
    )


def Z(res, sigma, k, alpha):
    try:
        k = k[0]
    except IndexError:
        pass
    z = numpy.ones(res.shape)
    term = res[res > 0] ** 2 / 2 / sigma - alpha * res[res > 0]
    z[res > 0] = k / (
        k + (1 - k) * numpy.sqrt(2 * numpy.pi * sigma) * alpha * numpy.exp(term)
    )
    return z


def sweep_row_mul(x, z):
    return (x.T * z).T


def apply_column_sum(x):
    return numpy.sum(x, axis=0)


def _flare_reg(
    x,
    y,
    k=None,
    beta=None,
    sigma=None,
    alpha=None,
    emg=1,
    epsilon=1e-4,
    maxit=1e4,
    verb=False,
    restart=50,
):

    print(x.shape)

    # Preamble
    x = numpy.concatenate((numpy.ones((x.shape[0], 1)), x), axis=1)
    
    n = len(y)
    p = x.shape[1]

    if k is None:
        kmean_fit = KMeans(n_clusters=2, random_state=None, init="k-means++").fit(
            y.reshape(-1, 1)
        )
        mem = kmean_fit.labels_
        y_one = y[mem == 0]
        y_two = y[mem == 1]

        k_one = len(y_one) / len(y)
        k_two = 1 - k_one
        k = numpy.array([k_one, k_two])[0]

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

    if alpha is None:
        lm__out = smf.ols(
            formula="y~x", data=pandas.DataFrame({"x": x[:, 1], "y": y})
        ).fit()
        a = 1 / numpy.sum(lm__out.resid[lm__out.resid > 0])
        alpha = numpy.abs(rng.normal(loc=a, size=(1,)))

    if emg:
        # try:
        res = emg_reg(x, y, intercept = True, beta=beta, sigma=sigma, k=alpha)
        # except:  # Unsafe, list which exceptions should be caught here.
        #     pass

        beta = res["beta"]
        sigma = res["sigma"]
        alpha = res["ex"]

    # Init
    diff = 1
    iter = 0
    counts = 0
    ll__counts = 0
    xbeta = (x @ beta.reshape(-1, 1)).squeeze()
    res = y - xbeta
    dn = spst.norm.pdf(res, scale=numpy.sqrt(sigma))
    de = spst.expon.pdf(res, scale=1 / alpha)
    ll = [loglik(res, sigma, k, alpha)]
    Q1 = -numpy.inf
    Q_all = None
    z = Z(res, sigma, k, alpha)

    while (numpy.sum(numpy.abs(diff) > epsilon) > 0) and iter < maxit:
        iter += 1
        res = y - (x @ beta.reshape(-1, 1)).squeeze()
        temp = (
            npla.pinv(-1 / sigma * x.T @ sweep_row_mul(x, z))
            @ (
                1 / sigma * apply_column_sum(sweep_row_mul(x, z * res))
                + alpha * apply_column_sum(sweep_row_mul(x, 1 - z))
            ).T
        )

        m = 1
        while m < restart:
            error = False
            try:
                beta__new = beta - temp
            except:  # Unsafe, list which exceptions should be caught here.
                error = True

            if error or numpy.isnan(beta__new).any():

                def Q_beta(beta_hat, sigma, k, alpha, z):
                    res_hat = y - (x @ beta_hat.reshape(-1, 1)).squeeze()
                    return Q(res_hat, sigma, k, alpha, z)

                result = spopt.minimize(
                    Q_beta,
                    beta,
                    args=(
                        sigma,
                        k,
                        alpha,
                        z,
                    ),
                )
                beta__new = result.x

            xbeta__new = (x @ beta__new.reshape(-1, 1)).squeeze()
            res__new = y - xbeta__new
            Q__beta = Q(res__new, sigma, k, alpha, z)
            z__new = Z(res__new, sigma, k, alpha)
            k__new = numpy.mean(z__new)
            sigma__new = numpy.sum(z__new * res__new ** 2) / numpy.sum(z__new)
            alpha__new = numpy.sum(1 - z__new[res__new > 0]) / numpy.sum(
                (1 - z__new[res__new > 0]) * res__new[res__new > 0]
            )
            # ugly for now
            diff = (
                numpy.r_[
                    [k__new] + beta__new.flatten().tolist() + [sigma__new, alpha__new]
                ]
                - numpy.r_[[k] + beta.flatten().tolist() + [sigma] + [alpha]]
            )

            z__new_two = Z(res__new, sigma__new, k__new, alpha__new)
            Q__new = Q(res__new, sigma__new, k__new, alpha__new, z__new_two)
            q__diff = Q__new - Q1
            if q__diff < 0:
                m += 1
            else:
                m = 101

        if m == restart:
            print("Too many attempts at step-halving!")

        k = k__new
        beta = beta__new
        xbeta = xbeta__new
        res = res__new
        sigma = sigma__new
        alpha = alpha__new
        z = z__new_two
        newobsloglik = loglik(res__new, sigma__new, k__new, alpha__new)
        ll.append(newobsloglik)
        counts += Q__new < Q1
        obsloglik = newobsloglik
        if verb:
            print(f"iteration = {iter}, diff = {diff}, log-likelihood = {obsloglik}")

    if iter == maxit:
        print("WARNING! NOT CONVERGENT!")
    print(f"number of iterations= {iter}")

    return mixEM(
        **{
            "x": x,
            "y": y,
            "posterior": numpy.concatenate(
                (z.reshape(-1, 1), (1 - z).reshape(-1, 1)), axis=1
            ),
            "k": [k, 1 - k],
            "beta": beta,
            "sigma": sigma,
            "alpha": alpha,
            "loglik": obsloglik,
            "all__loglik": ll,
            "ft": "flaremixEM",
        }
    )


def flare_reg(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    intercept: Optional[bool] = False,
    k: Optional[npt.ArrayLike] = None,
    beta: Optional[npt.ArrayLike] = None,
    sigma: Optional[float] = None,
    alpha: Optional[float] = None,
    emg: Optional[bool] = True,
    epsilon: Optional[float] = 1e-4,
    maxit: Optional[int] = 1e4,
    verb: Optional[bool] = False,
    restart: Optional[int] = 50,
):
    
    print(x.shape)
    print(y.shape)

    if len(numpy.array(x).squeeze().shape) == 1:
        x = numpy.atleast_2d(numpy.array(x)).reshape(-1,1)
        

    if intercept:
        x = x[..., 1:]

    y = numpy.array(y).reshape((-1,))

    return _flare_reg(
        x=x,
        y=y,
        k=k,
        beta=beta,
        sigma=sigma,
        alpha=alpha,
        emg=emg,
        epsilon=epsilon,
        maxit=maxit,
        verb=verb,
        restart=restart,
    )


if __name__ == "__main__":
    from emgregs import sim_flare_reg

    N = 500
    data = sim_flare_reg(xmin=1, xmax=7, n=N, beta=(2, 3, 5), sigma=0.1, alpha=1)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # axy = fig.add_subplot(111)
    # axy.plot(data["X"], data["Y"], "ko")
    # plt.show()

    regfit_flare = flare_reg(data["X"], data["Y"], maxit=10000)
