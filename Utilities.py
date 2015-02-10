import numpy as np


# Utilities
def machine_limit_exp(var, const, l=-700.0, h=700.0):
    var_mn = np.abs(var)
    var_mx = np.abs(var)
    const_mn = np.min(np.sign(var) * const)
    const_mx = np.max(np.sign(var) * const)
    if np.abs(var) * const_mn < l:
        var_mn = np.abs(l / const_mn)
    if np.abs(var) * const_mx > h:
        var_mx = np.abs(h / const_mx)
    return np.min([var_mn, var_mx, np.abs(var)]) * np.sign(var)
