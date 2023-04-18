import sys
import os

import h5py
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import palettable

from Template import RunningMean


plt.rcParams['figure.figsize'] = [16, 7]
plt.rc('font', family='Microsoft YaHei', size='18')

file_name = r"C:\Users\dzxmc\Downloads\ascadv2-extracted.h5"
file = h5py.File(file_name)

traces = file["Profiling_traces"]["traces"]
trace_mean_ = RunningMean()
for i in range(10000):
    sys.stdout.write("\rAdding trace {} to the average.\t".format(i))
    trace_mean_.update(traces[i])
trace_mean = trace_mean_()
del traces

print(trace_mean.shape)

fig = plt.figure(figsize=(18, 7), dpi=500)
plt.plot(trace_mean)
plt.xlabel("采样点")
plt.ylabel("能量消耗")
plt.ylim(-100, 100)
plt.axvline(1000, color="red")
plt.axvline(5000, color="red")
plt.axvline(5500, color="red")
plt.axvline(11000, color="red")
plt.axvline(12000, color="red")
plt.text(0, 75, "载入\n $r_{in}, r_{out}, r_m$")
plt.text(2500, 80, "计算GTab")
plt.text(5000, 75, "将 $state_M[perm[i]]$\n 转换为 $r_{in}$")
plt.text(8000, 80, "字节代替")
plt.text(11000, 75, "将 $r_{out}$ 转换为\n$state_M[perm[i]]$")
plt.text(13500, 80, "行移位")
plt.savefig("myMean.jpg")
plt.show()
