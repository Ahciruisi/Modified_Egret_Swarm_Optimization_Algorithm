# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from cec2017.functions import *
import math
import numpy as np
from collections import defaultdict

from ESOVectorised import ESO

# In[]
# data ={}
# for i in range(1, 31):
#     for j in range(4):
#         print(f"Function {i}, run {j}:")
#         func     = eval('f{}'.format(i))
#         n_dim = 30
#         population_size = 50
#         max_iter = 1200
#         lb   = np.array([-100]*n_dim)
#         ub   = np.array([100]*n_dim)
        
#         eso = ESO(func, n_dim, population_size, max_iter, lb, ub)
#         eso.run()
#         print('eso: ', eso.y_history[-1])
        
#         data['eso_f{}_{}'.format(i, j)]  = eso.y_history


# # In[]

# data = pd.DataFrame(data)

# data.to_csv('result/convergence/traditional_without_exploitation.csv')


# In[]
data_with_explore_only = pd.read_csv('result/convergence/traditional_without_exploitation.csv', index_col=0)
data_with_exploit_only = pd.read_csv('result/convergence/traditional_without_exploration.csv', index_col=0)

fitness_with_explore_only = data_with_explore_only.iloc[-1]
fitness_with_exploit_only = data_with_exploit_only.iloc[-1]

explore_only_fitness_avg = {}
exploit_only_fitness_avg = {}
explore_only_fitness_stddev = {}
exploit_only_fitness_stddev = {}
for i in range(1, 31):
    explore_only_fitnesses = np.array([fitness_with_explore_only[f"eso_f{i}_{j}"] for j in range(4)], dtype=float)
    explore_only_fitness_avg[f"f{i}"] = explore_only_fitnesses.mean()
    explore_only_fitness_stddev[f"f{i}"] = explore_only_fitnesses.std()

    exploit_only_fitnesses = np.array([fitness_with_exploit_only[f"eso_f{i}_{j}"] for j in range(4)], dtype=float)
    exploit_only_fitness_avg[f"f{i}"] = exploit_only_fitnesses.mean()
    exploit_only_fitness_stddev[f"f{i}"] = exploit_only_fitnesses.std()

count_better_means: "defaultdict[str, int]" = defaultdict(int)
count_better_stddevs: "defaultdict[str, int]" = defaultdict(int)
for key in explore_only_fitness_avg.keys():
    # print(key)
    # print("Averages:")
    # print(f"Exploration only: {explore_only_fitness_avg[key]}, Exploitation only: {exploit_only_fitness_avg[key]}")
    if (explore_only_fitness_avg[key] < exploit_only_fitness_avg[key]):
        count_better_means["Exploration only"] += 1
    elif (explore_only_fitness_avg[key] > exploit_only_fitness_avg[key]):
        count_better_means["Exploitation only"] += 1
    

    # print("STD Devs:")
    # print(f"Exploration only: {explore_only_fitness_stddev[key]}, Exploitation only: {exploit_only_fitness_stddev[key]}")
    if (explore_only_fitness_stddev[key] < exploit_only_fitness_stddev[key]):
        count_better_stddevs["Exploration only"] += 1
    elif (explore_only_fitness_stddev[key] > exploit_only_fitness_stddev[key]):
        count_better_stddevs["Exploitation only"] += 1

print("Better means count:")
print(count_better_means)
print("Better standard deviations count:")
print(count_better_stddevs)
# Better means count:
# defaultdict(<class 'int'>, {'Exploration only': 30})
# Better standard deviations count:
# defaultdict(<class 'int'>, {'Exploration only': 30})


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
    # print(key)
    # print("Averages:")
    # print(f"Without B: {without_b_fitness_avg[key]}, Without C: {without_c_fitness_avg[key]}")
    if (without_b_fitness_avg[key] < without_c_fitness_avg[key]):
        count_better_means["without B"] += 1
    elif (without_b_fitness_avg[key] > without_c_fitness_avg[key]):
        count_better_means["without C"] += 1
    

    # print("STD Devs:")
    # print(f"Without B: {without_b_fitness_stddev[key]}, Without C: {without_c_fitness_stddev[key]}")
    if (without_b_fitness_stddev[key] < without_c_fitness_stddev[key]):
        count_better_stddevs["without B"] += 1
    elif (without_b_fitness_stddev[key] > without_c_fitness_stddev[key]):
        count_better_stddevs["without C"] += 1

# Removing B gives lower mean and std-devs for almost all functions, so we should perhaps
# focus more on the aggressive random sampling strategy that C uses.
print("Better means count:")
print(count_better_means)
print("Better standard deviations count:")
print(count_better_stddevs)