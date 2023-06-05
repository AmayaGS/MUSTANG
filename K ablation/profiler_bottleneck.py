# -*- coding: utf-8 -*-
"""
Created on Fri May  5 01:26:26 2023

@author: AmayaGS
"""

import torchsummary

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from loaders import Loaders

from graph_train_loop import train_graph_slides, train_graph_multi_stain

from clam_model import VGG_embedding, GatedAttention
from Graph_model import GAT_TopK, GAT_SAGPool, GAT_SAGPool_flops_1, GCN_topK, GCN_SAGPool, GAT_model, GCN_model

from plotting_results import auc_plot, pr_plot, plot_confusion_matrix
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

import gc 
gc.enable()

# %%

graph_net = GAT_SAGPool_flops_1(1024, heads=1)
graph_net.cuda()

input_data = torch.randn(2000, 1024) 

# %%

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, with_flops=True) as prof:
    #with record_function("model_inference"):
    graph_net(input_data)

# %%

print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))

#print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=20))

# %%

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    graph_net(input_data)

prof.export_chrome_trace("trace.json")

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# %%

df = pd.read_csv(r"C:\Users\Amaya\Documents\PhD\Conferences\BMVC 2023\MUSTANG\ram_flops_time.csv")

# %%

fig, ax1 = plt.subplots(figsize=(15,15))
ax1.plot(df['K'], df['GPU RAM [GB]'], color='darkred')
ax2 = ax1.twinx()
ax2.plot(df['K'], df['MFLOPs'], color='darkblue')
trend = np.polyfit(df['MFLOPs'], df['K'], 1)
fit = np.poly1d(trend)
ax2.plot(fit(df['MFLOPs']), df['MFLOPs'], "r--")
ax1.set_ylabel('GPU RAM [GB]', size=30, labelpad=20)
ax2.set_ylabel('FLOPs [M]', size=30, labelpad=20)
ax1.tick_params(axis='y', labelcolor='darkblue', labelsize=25)
ax2.tick_params(axis='y', labelcolor='darkred', labelsize=25)
ax1.tick_params(axis='x', labelsize=25)
#ax1.set_xticklabels(df['K'], fontsize=15)
#ax2.set_yticklabels(df['MFLOPs'], fontsize=12)
ax1.set_xlabel('K', size=30, labelpad=25)
plt.show()


# %%

trend = np.polyfit(df['MFLOPs'], df['K'], 2)
fit = np.poly1d(trend)
plt.plot(fit(df['MFLOPs']), df['MFLOPs'], "r--")
plt.show()

results = {}
results['polynomial'] = trend.tolist()
correlation = np.corrcoef(df['MFLOPs'],  df['K'])[0,1]
results['correlation'] = correlation
results['rsquared'] = correlation**2

# %%

p = np.poly1d(trend)
# fit values, and mean
yhat = p(df['MFLOPs'])                         # or [p(z) for z in x]
ybar = np.sum(df['K'])/len(df['K'])          # or sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((df['K'] - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
results['determination'] = ssreg / sstot
#label='Sensitivity = %0.2f\nSpecificity = %0.2f' % (sensitivity, specificity)

# %%

plt.figure(figsize=(15,15))
plt.plot(df['K'], df['time [ms]'], color='darkred')
trend = np.polyfit(np.log((df['time [ms]'])), np.log(df['K']), 1, w=np.sqrt(df['K']) )
fit = np.poly1d(trend)
plt.plot(fit(np.sort(np.log(df['time [ms]']))), np.log(np.sort(df['time [ms]'])), "r--")
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
#ax2.set_ylabel('FLOPs [M]', size=20, labelpad=20)
#plt.title('Average Precision = %0.2f' % auc_precision_recall, size=20)
plt.ylabel('Foward pass compute time [ms]', size=30, labelpad=20)
plt.xlabel('K', size=30, labelpad=20)
#plt.legend(loc='lower left', prop={'size': 19})
plt.show()

# %%

from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline, BSpline
from scipy.interpolate import interp1d

def func(x, a, b, c):
  #return a * np.exp(-b * x) + c
  return a * np.log(b * x) + c

x = df['K']   # changed boundary conditions to avoid division by 0
y = df['time [ms]']
yn = y + 0.2*np.random.normal(size=len(x))

popt, pcov = curve_fit(func, x, yn) 


plt.figure(figsize=(15,12))
plt.plot(x, y, color='darkblue')
plt.plot(x, func(x, *popt), 'r--')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel('Foward pass compute time [ms]', size=30, labelpad=20)
plt.xlabel('K', size=30, labelpad=20)
#plt.legend()
plt.show()