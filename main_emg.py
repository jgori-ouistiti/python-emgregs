import function_simulation as sim
import function_emg__reg as reg
import function_try__flare as flare

N = 500
data = sim.sim__emg_reg(n=N, sigma=1, alpha=0.1)
regfit_emg = reg.func_emg__reg(data["X"], data["Y"], maxit=10000)
regfit_flare = flare.fun_try__flare__reg(data["Y"], data["X"], maxit=10000)
