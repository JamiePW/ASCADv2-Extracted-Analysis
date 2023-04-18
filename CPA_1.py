import sys
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import palettable

from Template import permIndices, CPA, CPA_2o, CPA_3o

ATTACK_N_A = 100000

plt.rcParams['figure.figsize'] = [16, 4]
plt.rc('font', family='SimHei', weight='bold', size='20')

# 将全部16个字节的PoI范围整合起来，以便在不知乱序变换时用于攻击
all_c_range = np.arange(5550, 5561)
for i in range(1, 16):
    all_c_range = np.append(all_c_range, np.arange(5550 + 373*i, 5561 + 373*i))

file_name = r"C:\Users\dzxmc\Downloads\ascadv2-extracted.h5"
file = h5py.File(file_name)

c_pick = [5552, 5553]
traces = file["Profiling_traces"]["traces"][0:ATTACK_N_A, c_pick]    # 包含c[0]泄露的能量迹片段

# traces = file["Profiling_traces"]["traces"][0:ATTACK_N_A, all_c_range]
# traces = traces.reshape(ATTACK_N_A, 11, -1).mean(axis=2)
print("shape of traces: " + str(traces.shape))

meta = file["Profiling_traces"]["metadata"][0:ATTACK_N_A]
labels = file["Profiling_traces"]["labels"][0:ATTACK_N_A]
masks = meta["masks"][0:ATTACK_N_A]

inds = np.array([permIndices(np.arange(16), *masks[i][:4]) for i in range(ATTACK_N_A)])[:, 0]
print("shape of inds: " + str(inds.shape))

plains = meta["plaintext"][:, 0]
keys = meta["key"][:, 0]

# plains = np.array([meta["plaintext"][i, inds[i]] for i in range(ATTACK_N_A)])
# keys = np.array([meta["key"][i, inds[i]] for i in range(ATTACK_N_A)])

alphas = labels["alpha_mask"].flatten()
betas = labels["beta_mask"].flatten()

print("shape of plains: " + str(plains.shape) + " shape of keys: " + str(keys.shape) + " shape of alphas: " + str(alphas.shape) + " shape of betas: " + str(betas.shape))

my_CPA_ = CPA(traces=traces, plains=plains, keys=keys, alphas=alphas, betas=betas)

ge, succ_rate, corr_k_t = my_CPA_.compute_ge()

print(ge)
print(succ_rate)
print(corr_k_t[-10:])


'''plt.subplot(2, 1, 1, figsize=(15, 7))
plt.plot(ge, label="Guess Entropy")
plt.legend()

plt.subplot(2, 1, 2, figsize=(15, 7))
plt.plot(succ_rate, label="Success Rate")
plt.legend()

plt.show()'''

fig, ax = plt.subplots(figsize=(15, 7), dpi=500)
plt.yscale('log', base=2)
plt.ylim(0.4, 600)
# ax.set_title("Guess Entropy and Success Rate under 1st order CPA (known permutation, known rm, rout)")
curr_Na = np.arange(ge.shape[0])
ax.set_xlabel("能量迹数量")

ax.set_ylabel("猜测熵")
line0 = ax.plot(curr_Na, ge, label="Guess Entropy", color="red")

ax_right = ax.twinx()
ax_right.set_ylabel("成功率")
ax_right.set_ylim(-0.1, 1.1)
line1 = ax_right.plot(curr_Na, succ_rate, label="Success Rate", color="blue")

# ax.axvline(530, color="black")

plt.legend([line0[0], line1[0]], ["猜测熵", "成功率"], fancybox=True, loc="center right")
# plt.savefig("v2ex_1o_no_perm_rm_rout.jpg")
plt.show()




fig, ax = plt.subplots(figsize=(15, 7), dpi=500)
# ax.set_title("Maximum value of the correlation coefficient under 1st order CPA (unknown permutation, known rm, rout)")
ax.set_ylim(0, 1.0)
ax.set_xlabel("能量迹数量")
ax.set_ylabel("相关系数")
ax.plot(corr_k_t, label="相关系数$\\rho_{ck,tk}$")
plt.legend(fancybox=True)
plt.savefig("corr_v2ex_1o_no_perm_rm_rout.jpg")
plt.show()

