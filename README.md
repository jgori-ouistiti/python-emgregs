# emgregs: Exponentially Modified Gaussian (EMG) Noise REGressionS

> code to perform EMG regression and EMG regression with flare

The EMG regression with and without flare are presented in [Gori et al. 2019](https://hal.archives-ouvertes.fr/hal-02191051/document) and [Li et al., in preparation](https://locallhost.com/) respectively.

You can get this package from Pypi: 

```shell
python3 -m pip install emgregs
```

The package is supported for Python 3.8 and above


Example usage:

```python
func_emg__reg(x, y)
fun_try__flare__reg(y, x, maxit=10000)
```
