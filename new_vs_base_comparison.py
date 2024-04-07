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
from ESOmodified import ESOModified

# In[]
eso_data ={}
eso_modified_data ={}
for i in range(1, 31):
    for j in range(10):
        print(f"Function {i}, run {j}:")
        func     = eval('f{}'.format(i))
        n_dim = 30
        population_size = 50
        max_iter = 1200
        lb   = np.array([-100]*n_dim)
        ub   = np.array([100]*n_dim)
        
        eso = ESO(func, n_dim, population_size, max_iter, lb, ub)
        eso.run()
        print('eso: ', eso.y_history[-1])
        
        eso_data['eso_f{}_{}'.format(i, j)]  = eso.y_history

        eso_modified = ESOModified(func, n_dim, population_size, max_iter, lb, ub)
        eso_modified.run()
        print('modified: ', eso_modified.y_history[-1])
        
        eso_modified_data['modified_f{}_{}'.format(i, j)]  = eso_modified.y_history


# In[]

eso_data = pd.DataFrame(eso_data)
eso_modified_data = pd.DataFrame(eso_modified_data)

eso_data.to_csv('result/modifications/cec17_fitness_eso.csv')
eso_modified_data.to_csv('result/modifications/cec17_fitness_modified.csv')


# In[]
eso_data = pd.read_csv('result/modifications/cec17_fitness_eso.csv', index_col=0)
eso_modified_data = pd.read_csv('result/modifications/cec17_fitness_modified.csv', index_col=0)

eso_fitness = eso_data.iloc[-1]
eso_modified_fitness = eso_modified_data.iloc[-1]

error_diffs = (eso_fitness - eso_modified_fitness).map(lambda x: math.copysign(1, x))
error_diffs = error_diffs.map(lambda signum: {1: "Modified", 0: "same", -1: "ESO"}.get(signum))
count_of_errors_for_each_algo = np.unique(error_diffs.to_numpy(), return_counts=True)

better_algo, error_count_without_egret = np.min(count_of_errors_for_each_algo, axis=1)
print(f"Found better result with optimiser {better_algo}: {error_count_without_egret}")

eso_fitness_avg = {}
eso_modified_fitness_avg = {}
eso_fitness_stddev = {}
eso_modified_fitness_stddev = {}
for i in range(1, 31):
    eso_fitnesses = np.array([eso_fitness[f"eso_f{i}_{j}"] for j in range(5)], dtype=float)
    eso_fitness_avg[f"f{i}"] = eso_fitnesses.mean()
    eso_fitness_stddev[f"f{i}"] = eso_fitnesses.std()

    eso_modified_fitnesses = np.array([eso_modified_fitness[f"modified_f{i}_{j}"] for j in range(5)], dtype=float)
    eso_modified_fitness_avg[f"f{i}"] = eso_modified_fitnesses.mean()
    eso_modified_fitness_stddev[f"f{i}"] = eso_modified_fitnesses.std()

count_better_means: "defaultdict[str, int]" = defaultdict(int)
count_better_stddevs: "defaultdict[str, int]" = defaultdict(int)
for key in eso_fitness_avg.keys():
    print(key)
    print("Averages:")
    print(f"ESO: {eso_fitness_avg[key]}, MESO: {eso_modified_fitness_avg[key]}")
    if (eso_fitness_avg[key] < eso_modified_fitness_avg[key]):
        count_better_means["eso"] += 1
    elif (eso_fitness_avg[key] > eso_modified_fitness_avg[key]):
        count_better_means["modified"] += 1
    

    print("STD Devs:")
    print(f"ESO: {eso_fitness_stddev[key]}, MESO: {eso_modified_fitness_stddev[key]}")
    if (eso_fitness_stddev[key] < eso_modified_fitness_stddev[key]):
        count_better_stddevs["eso"] += 1
    elif (eso_fitness_stddev[key] > eso_modified_fitness_stddev[key]):
        count_better_stddevs["modified"] += 1

print("Better means count:")
print(count_better_means)
print("Better standard deviations count:")
print(count_better_stddevs)

# print("ESO:")
# print("Average:")
# print()

# eso_data ={}
# eso_modified_data ={}
# for i in range(1, 31):
#     for j in range(5):
#         print(f"Function {i}, run {j}:")
#         func     = eval('f{}'.format(i))
#         n_dim = 30
#         population_size = 50
#         max_iter = 500
#         lb   = np.array([-100]*n_dim)
#         ub   = np.array([100]*n_dim)
        
#         eso_modified = ESOModified(func, n_dim, population_size, max_iter, lb, ub)
#         eso_modified.run()
#         print('modified: ', eso_modified.y_history[-1])
        
#         eso_modified_data['modified_f{}_{}'.format(i, j)]  = eso_modified.y_history


# # In[]

# eso_modified_data = pd.DataFrame(eso_modified_data)
# eso_modified_data.to_csv('result/modifications/cec17_fitness_modified_test.csv')