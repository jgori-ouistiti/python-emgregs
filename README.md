# emgregs: Exponentially Modified Gaussian (EMG) Noise REGressionS

> code to perform EMG regression and EMG regression with flare

The EMG regression with and without flare are presented in [Gori et al. 2019](https://hal.archives-ouvertes.fr/hal-02191051/document) and [Li et al., in preparation](https://locallhost.com/) respectively.

You can get this package from Pypi: 

```shell
python3 -m pip install emgregs
```

The package is supported for Python 3.8 and above.


# Example usage:

```python

from emgregs import sim_emg_reg, sim_emg_reg_heterosked, sim_flare_reg, emg_reg, emg_reg_heterosked, flare_reg

## Generate data following EMG or flare model
N = 5000
emg_data = sim_emg_reg(
xmin=1, xmax=7, n=N, beta=(2, 3, 5), sigma=0.5, alpha=0.01
)
emg_data_heterosked = sim_emg_reg_heterosked(
xmin=1, xmax=7, n=N, beta=(2, 3, 5), sigma=0.1, expo_scale=(0.1, 0.1, 0.1)
)
flare_data = sim_flare_reg(xmin=1, xmax=7, n=N, beta=(2, 3, 5), sigma=0.1, alpha=1)

## Compute EMG regression
emg_homosked = emg_reg(emg_data["X"], emg_data["Y"], maxit=10000)
emg_heterosked = emg_reg_heterosked(emg_data_heterosked["X"], emg_data_heterosked["Y"], maxit=10000)
    
## Compute Flare regression
flare = flare_reg(flare_data["Y"], flare_data["X"], maxit=10000)

```

# Model:

$$Y = \beta X + Z$$

For EMG, Z = E + G, where E is exponential and N is Gaussian. For flare, Z = $\lambda$ E + (1-$\lambda$) G (mixture model)


# Parameters:

* xmin, xmax: domain of covariates
* n: sample size
* beta: covariate weights
* sigma: standard deviation of G component
* alpha, k: rate of E
* expo_scale: scale of E (only in heterosked mode)
