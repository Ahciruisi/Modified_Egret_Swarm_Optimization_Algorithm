# -*- coding: utf-8 -*-
# In[]
import numpy as np
import pandas as pd

# In[]

data = pd.read_csv('result/modifications/Himmelblau_solution.csv')
otherData = pd.read_csv('result/modifications/Himmelblau_solution_others.csv')

meso_solution = []
eso_solution = []
pso_solution = []
ga_solution  = []
de_solution  = []
# gwo_solution = []
# hho_solution = []

no_of_runs = 15

for i in range(no_of_runs):
    meso_solution.append(data['Himmelblau_solution_{}'.format(i)].values[-1])
    eso_solution.append(otherData['Himmelblau_solution_eso_{}'.format(i)].values[-1])
    # pso_solution.append(otherData['Himmelblau_solution_pso_{}'.format(i)].values[-1])
    # ga_solution.append(otherData['Himmelblau_solution_ga_{}'.format(i)].values[-1])
    # de_solution.append(otherData['Himmelblau_solution_de_{}'.format(i)].values[-1])

meso_solution = np.array(meso_solution)
eso_solution = np.array(eso_solution)
# pso_solution = np.array(pso_solution)
# ga_solution  = np.array(ga_solution)
# de_solution  = np.array(de_solution)

print("Himmelblau:")
print('meso:', meso_solution.min(), meso_solution.max(), meso_solution.mean(), meso_solution.std())
print('eso: ', eso_solution.min(), eso_solution.max(), eso_solution.mean(), eso_solution.std())
# print('pso: ', pso_solution.min(), pso_solution.max(), pso_solution.mean(), pso_solution.std())
# print('ga:  ', ga_solution.min(), ga_solution.max(), ga_solution.mean(), ga_solution.std())
# print('de:  ', de_solution.min(), de_solution.max(), de_solution.mean(), de_solution.std())



# In[]



data = pd.read_csv('result/modifications/Spring_solution.csv')
otherData = pd.read_csv('result/modifications/Spring_solution_others.csv')

meso_solution = []
eso_solution = []
# pso_solution = []
# ga_solution  = []
# de_solution  = []

for i in range(no_of_runs):
    meso_solution.append(data['spring_solution_{}'.format(i)].values[-1])
    eso_solution.append(otherData['spring_solution_eso_{}'.format(i)].values[-1])
    # pso_solution.append(otherData['spring_solution_pso_{}'.format(i)].values[-1])
    # ga_solution.append(otherData['spring_solution_ga_{}'.format(i)].values[-1])
    # de_solution.append(otherData['spring_solution_de_{}'.format(i)].values[-1])

meso_solution = np.array(meso_solution)
eso_solution = np.array(eso_solution)
# pso_solution = np.array(pso_solution)
# ga_solution  = np.array(ga_solution)
# de_solution  = np.array(de_solution)

print("Spring:")
print('meso:', meso_solution.min(), meso_solution.max(), meso_solution.mean(), meso_solution.std())
print('eso: ', eso_solution.min(), eso_solution.max(), eso_solution.mean(), eso_solution.std())
# print('pso: ', pso_solution.min(), pso_solution.max(), pso_solution.mean(), pso_solution.std())
# print('ga:  ', ga_solution.min(), ga_solution.max(), ga_solution.mean(), ga_solution.std())
# print('de:  ', de_solution.min(), de_solution.max(), de_solution.mean(), de_solution.std())







