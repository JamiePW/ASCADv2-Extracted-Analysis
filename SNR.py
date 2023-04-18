import sys
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import palettable

from Template import permIndices, RunningSNR, multGF256, Sbox, RunningMean

plt.rcParams['figure.figsize'] = [16, 4]
plt.rc('font', family='Microsoft YaHei', size='16')

file_name = r"C:\Users\dzxmc\Downloads\ascadv2-extracted.h5"
file = h5py.File(file_name)

traces = file["Profiling_traces"]["traces"]
meta = file["Profiling_traces"]["metadata"]


def get_data(i, byte=0):
    """
    从HDF5文件中获取数据
    """
    trace = traces[i]
    mask = meta[i]["masks"]
    key, plain = meta[i]["plaintext"], meta[i]["key"]
    s = Sbox[key ^ plain]
    inds = permIndices(np.arange(16), *mask[:4])
    r_in, r_out, r_m = mask[16], mask[17], mask[18]
    c = multGF256(r_m, s[inds[byte]]) ^ r_out
    c1 =  multGF256(r_m, s[inds[byte]])
    c2 = s[inds[byte]]
    return trace, inds, r_in, r_out, r_m, c, c1, c2


SNR_inds_ = RunningSNR(n_classes=16)    # 乱序变换索引
SNR_r_in_ = RunningSNR(n_classes=256)   # 布尔掩码rin
SNR_r_out_ = RunningSNR(n_classes=256)  # 布尔掩码rout
SNR_r_m_ = RunningSNR(n_classes=255)    # 乘法掩码rm
SNR_c_ = RunningSNR(n_classes=256)      # 仿射掩码保护的中间状态值
SNR_c1_ = RunningSNR(n_classes=256)     # 乘法掩码保护的中间状态值
SNR_c2_ = RunningSNR(n_classes=256)     # 无掩码保护的中间状态值
trace_mean_ = RunningMean()             # 能量迹


for i in range(10000):
    trace, inds, r_in, r_out, r_m, c, c1, c2 = get_data(i, byte=0)
    sys.stdout.write("\r{}\t".format(i))

    SNR_inds_.update(trace, inds[0])
    SNR_r_in_.update(trace, r_in)
    SNR_r_out_.update(trace, r_out)
    SNR_r_m_.update(trace, r_m - 1)
    SNR_c_.update(trace, c)
    SNR_c1_.update(trace, c1)
    SNR_c2_.update(trace, c2)
    trace_mean_.update(trace)

    del trace, inds, r_in, r_out, r_m, c, c1, c2

SNR_inds = SNR_inds_()
SNR_r_in = SNR_r_in_()
SNR_r_out = SNR_r_out_()
SNR_r_m = SNR_r_m_()
SNR_c = SNR_c_()
SNR_c1 = SNR_c1_()
SNR_c2 = SNR_c2_()
trace_mean = trace_mean_()

target_SNR = [SNR_inds, SNR_r_in, SNR_r_out, SNR_r_m, SNR_c, SNR_c1, SNR_c2]
target_str = ["$permIndices$", "$r_{in}$", "$r_{out}$", "$r_{m}$", "$c[0]$", "$r_m*Z[perm[i]]$", "$Z[perm[i]]$"]

for i in range(7):
    fig, ax = plt.subplots(figsize=(15, 7), dpi=500)
    trace_x = np.arange(trace_mean.shape[0])
    ax.set_xlabel("采样点")
    # ax.set_title(f"关于{target_str[i]}的信噪比")
    ax.set_ylabel("能量消耗")
    ax.set_ylim(-100, 100)
    line0 = ax.plot(trace_x, trace_mean, alpha=0.9, label="Power consumption trace", color="blue")
    ax_right = ax.twinx()
    ax_right.set_ylabel("信噪比")
    line1 = ax_right.plot(trace_x, target_SNR[i], alpha=0.9, label=f"SNR targeting {target_str[i]}", color="red")
    plt.legend([line0[0], line1[0]], ["能量迹", f"关于{target_str[i]}的信噪比"], fancybox=True, loc="upper right")
    plt.savefig(f"SNR_{i}.jpg")
    plt.show()

'''
注意！！！
Sbox16个字节之间，基本相差373个点
'''

max_SNR_c = max(SNR_c)
print(max_SNR_c)
print(np.where(SNR_c > (max_SNR_c / 1.5)))

max_SNR_rout = max(SNR_r_out)
print(max_SNR_rout)
print(np.where(SNR_r_out > (max_SNR_rout / 1.5)))

max_SNR_rm = max(SNR_r_m)
print(max_SNR_rm)
print(np.where(SNR_r_m >= (max_SNR_rm / 1.4)))

