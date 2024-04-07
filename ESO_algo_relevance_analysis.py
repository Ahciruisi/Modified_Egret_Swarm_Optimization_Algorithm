# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 19:59:17 2022

@author: 山抹微云
"""

from testFunc import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
from cec2017.functions import *
import math
import numpy as np
from collections import defaultdict

from ESOnew import ESO

# In[]
# data ={}
# for i in range(1, 31):
#     for j in range(4):
#         print(f"Function {i}, run {j}:")
#         func     = eval('f{}'.format(i))
#         n_dim = 30
#         population_size = 50
#         max_iter = 2000
#         lb   = np.array([-100]*n_dim)
#         ub   = np.array([100]*n_dim)
        
#         eso = ESO(func, n_dim, population_size, max_iter, lb, ub)
#         eso.run()
#         print('eso: ', eso.y_history[-1])
        
#         data['eso_f{}_{}'.format(i, j)]  = eso.y_history


# # # In[]

# data = pd.DataFrame(data)

# data.to_csv('result/convergence/traditional_without_egret_c.csv')


# In[]
data_without_b = pd.read_csv('result/convergence/traditional_without_egret_b.csv', index_col=0)
data_without_c = pd.read_csv('result/convergence/traditional_without_egret_c.csv', index_col=0)

fitness_without_b = data_without_b.iloc[-1]
fitness_without_c = data_without_c.iloc[-1]

without_b_fitness_avg = {}
without_c_fitness_avg = {}
without_b_fitness_stddev = {}
without_c_fitness_stddev = {}
for i in range(1, 31):
    without_b_fitnesses = np.array([fitness_without_b[f"eso_f{i}_{j}"] for j in range(4)], dtype=float)
    without_b_fitness_avg[f"f{i}"] = without_b_fitnesses.mean()
    without_b_fitness_stddev[f"f{i}"] = without_b_fitnesses.std()

    without_c_fitnesses = np.array([fitness_without_c[f"eso_f{i}_{j}"] for j in range(4)], dtype=float)
    without_c_fitness_avg[f"f{i}"] = without_c_fitnesses.mean()
    without_c_fitness_stddev[f"f{i}"] = without_c_fitnesses.std()

count_better_means: "defaultdict[str, int]" = defaultdict(int)
count_better_stddevs: "defaultdict[str, int]" = defaultdict(int)
for key in without_b_fitness_avg.keys():
    print(key)
    print("Averages:")
    print(f"Wihtout B: {without_b_fitness_avg[key]}, Without C: {without_c_fitness_avg[key]}")
    if (without_b_fitness_avg[key] < without_c_fitness_avg[key]):
        count_better_means["without B"] += 1
    elif (without_b_fitness_avg[key] > without_c_fitness_avg[key]):
        count_better_means["without C"] += 1
    

    print("STD Devs:")
    print(f"Wihtout B: {without_b_fitness_stddev[key]}, Without C: {without_c_fitness_stddev[key]}")
    if (without_b_fitness_stddev[key] < without_c_fitness_stddev[key]):
        count_better_stddevs["without B"] += 1
    elif (without_b_fitness_stddev[key] > without_c_fitness_stddev[key]):
        count_better_stddevs["without C"] += 1

# Removing B gives lower mean and std-devs for almost all functions.
print("Better means count:")
print(count_better_means)
print("Better standard deviations count:")
print(count_better_stddevs)