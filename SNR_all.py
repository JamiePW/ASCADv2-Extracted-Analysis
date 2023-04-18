import sys
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import palettable

from Template import permIndices, RunningSNR, multGF256, Sbox, RunningMean

plt.rcParams['figure.figsize'] = [16, 4]
plt.rc('font', family='Microsoft YaHei', size='18')

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
    return trace, inds, r_in, r_out, r_m, c


SNR_inds_ = RunningSNR(n_classes=16)
SNR_r_in_ = RunningSNR(n_classes=256)
SNR_r_out_ = RunningSNR(n_classes=256)
SNR_r_m_ = RunningSNR(n_classes=255)
SNR_c_ = RunningSNR(n_classes=256)
trace_mean_ = RunningMean()


for i in range(10000):
    trace, inds, r_in, r_out, r_m, c = get_data(i, byte=0)
    sys.stdout.write("\r{}\t".format(i))

    SNR_inds_.update(trace, inds[0])
    SNR_r_in_.update(trace, r_in)
    SNR_r_out_.update(trace, r_out)
    SNR_r_m_.update(trace, r_m - 1)
    SNR_c_.update(trace, c)
    trace_mean_.update(trace)

    del trace, inds, r_in, r_out, r_m, c

SNR_inds = SNR_inds_()
SNR_r_in = SNR_r_in_()
SNR_r_out = SNR_r_out_()
SNR_r_m = SNR_r_m_()
SNR_c = SNR_c_()
trace_mean = trace_mean_()

fig=plt.figure(figsize=(15, 6), dpi=500)
plt.xlabel("采样点")
plt.ylabel("信噪比")
plt.plot(SNR_inds, alpha=0.9, label="permIndices")
plt.plot(SNR_r_in, alpha=0.9, label="$r_{in}$")
plt.plot(SNR_r_out, alpha=0.9, label="$r_{out}$")
plt.plot(SNR_r_m, alpha=0.9, label="$r_m$")
plt.plot(SNR_c, alpha=0.9, label="$c[0]$")
plt.legend()
plt.savefig("SNR.jpg")
plt.show()

