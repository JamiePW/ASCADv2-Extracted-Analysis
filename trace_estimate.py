import numpy as np


# z_0.9999 = 3.719
def trace_estimate(corr_k_t, z_a=3.719):
    temp = np.log((1 + corr_k_t) / (1 - corr_k_t))
    temp = temp ** 2
    n = 3 + 8 * (z_a ** 2 / temp)
    return n


corr_k_t = 0.05100

print(trace_estimate(corr_k_t=corr_k_t))
