from emgregs import sim_emg_reg, sim_emg_reg_heterosked, sim_flare_reg, emg_reg, emg_reg_heterosked, flare_reg, __version__

import numpy

def test_version():
    assert __version__ == "0.0.6-dev0"

def test_with_sim_emg():
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
    flare = flare_reg(flare_data["X"], flare_data["Y"], maxit=10000)

def test_with_arrays():
    ## Generate data following EMG or flare model
    N = 5000
    x = numpy.linspace(2,8, N)
    y = 0.1 + 0.1*x + numpy.random.normal(0, 1, size = x.shape)

    ## Compute EMG regression
    emg_homosked = emg_reg(x, y, maxit=10000)
    emg_heterosked = emg_reg_heterosked(x, y, maxit=10000)
        
    ## Compute Flare regression
    flare = flare_reg(x, y, maxit=10000)

def test():
    test_with_sim_emg()
    test_with_arrays()

if __name__ == "__main__":
    test()