# -*- coding: utf-8 -*-
import numpy as np
from ESO import ESO
from ESOModified import ESOModified

import pandas as pd
import matplotlib.pyplot as plt


# In[] Compression String Design Problem 


def f_String(x):
    temp = 0
    sig  = 1e5
    constraint_ueq = [
        lambda x: 1 - x[1]**3*x[2]/(71785*x[0]**4),
        lambda x: (4*x[1]**2-x[0]*x[1])/(12566*(x[1]*x[0]**3-x[0]**4)) + 1/(5108*x[0]**2)-1,
        lambda x: 1-140.45*x[0]/(x[1]**2*x[2]),
        lambda x: (x[0]+x[1])/1.5-1
        ]
    for constraint in constraint_ueq:
        if constraint(x)>0:
            temp+=constraint(x)**2*sig
    return (x[2]+2)*x[1]*x[0]**2+temp

# In[]
no_of_runs = 30

n_dim = 3
size_pop = 10
max_iter = 500


lb = np.array([0.05, 0.25, 2.00])
ub = np.array([2.00, 1.30, 15.0])

# In[]
data = {}
meso_solution = []
meso_x = []
for i in range(no_of_runs):
    meso = ESOModified(f_String, n_dim, size_pop, max_iter, lb, ub)
    meso.run()
    meso_x.append(meso.x_global_best)
    data['spring_solution_{}'.format(i)] = meso.y_history
    meso_solution.append(meso.y_global_best)
    print('eso modified: ', meso.y_global_best)
meso_solution = np.array(meso_solution)

print(meso_solution.mean(), meso_solution.std())
data = pd.DataFrame(data)
data.to_csv('result/modifications/Spring_solution.csv')

x_data = {'x_best': meso_x[meso_solution.argmin()], 
          'f_best': meso_solution.min(), 
          'f_worst':meso_solution.max(),
          'f_ave': meso_solution.mean(),
          'f_std': meso_solution.std()}
x_data = pd.DataFrame(x_data)
x_data.to_csv('result/modifications/Spring_result.csv')

# In[]
from sko.PSO import PSO
from sko.DE  import DE
from sko.GA  import GA
# In[]
data = {}

pso_solution = []
ga_solution = []
de_solution = []
eso_solution = []
for i in range(no_of_runs):
    
    pso = PSO(func=f_String, dim=n_dim, pop=size_pop, max_iter=max_iter, lb=lb, ub=ub, w=0.8, c1=0.5, c2=0.5)
    pso.run()
    data['spring_solution_pso_{}'.format(i)] = pso.gbest_y_hist
    pso_solution.append(pso.gbest_y_hist[-1])
    print('pso: ', pso.gbest_y_hist[-1])
    
    de = DE(func=f_String, n_dim=n_dim, size_pop=size_pop, max_iter=max_iter, lb=lb, ub=ub)
    de.run()
    data['spring_solution_de_{}'.format(i)] = de.generation_best_Y
    de_solution.append(de.generation_best_Y[-1])
    print('de: ', de.generation_best_Y[-1])
    
    
    ga = GA(func=f_String, n_dim=n_dim, size_pop=size_pop, max_iter=max_iter, prob_mut=0.001, lb=lb, ub=ub, precision=1e-7)
    ga.run()
    data['spring_solution_ga_{}'.format(i)] = ga.generation_best_Y
    ga_solution.append(ga.generation_best_Y[-1])
    print('ga: ', ga.generation_best_Y[-1])
    
    eso = ESO(f_String, n_dim, size_pop, max_iter, lb, ub)
    eso.run()
    data['spring_solution_eso_{}'.format(i)] = eso.y_history
    eso_solution.append(eso.y_global_best)
    

data = pd.DataFrame(data)
data.to_csv('result/modifications/Spring_solution_others.csv')

    
# In[]
def f_Himmelblau(x):
    def g1(x):
        return 85.334407 + 0.0056858*x[1]*x[4] + 0.0006262*x[0]*x[3] - 0.0022053*x[2]*x[4]

    def g2(x):
        return 80.51249 + 0.0071317*x[1]*x[4] + 0.0029955*x[0]*x[1] + 0.0021813*x[2]**2

    def g3(x):
        return 9.300961 + 0.0047026*x[2]*x[4] + 0.0012547*x[0]*x[2] + 0.0019085*x[2]*x[3]

    constraint_ueq = [
        lambda x: g1(x) - 92,
        lambda x: -g1(x),
        lambda x: g2(x) - 110,
        lambda x: 90 - g2(x),
        lambda x: g3(x) - 25,
        lambda x: 20 - g3(x)
        ]
    
    temp = 0
    sig  = 1e100
    for constraint in constraint_ueq:
        if constraint(x)>0:
            temp+=constraint(x)**2*sig
    return 5.3578547*x[2]**2 + 0.8356891*x[0]*x[4]+37.293239*x[0]-40792.14+temp






n_dim = 5
size_pop = 10
max_iter = 500
lb = np.array([78 , 33, 27, 27, 27])
ub = np.array([102, 45, 45, 45, 45])

# In[]

data = {}
meso_solution = []
meso_x = []
for i in range(no_of_runs):
    meso = ESOModified(f_Himmelblau, n_dim, size_pop, max_iter=max_iter, lb=lb, ub=ub)
    meso.run()
    print('eso modified: ', meso.y_global_best)
    data['Himmelblau_solution_{}'.format(i)] = meso.y_history
    meso_solution.append(meso.y_global_best)
    meso_x.append(meso.x_global_best)
    
meso_solution = np.array(meso_solution)    
data = pd.DataFrame(data)
data.to_csv('result/modifications/Himmelblau_solution.csv')
meso_x = {'x_best': meso_x[meso_solution.argmin()], 
          'f_best': meso_solution.min(), 
          'f_worst':meso_solution.max(),
          'f_ave': meso_solution.mean(),
          'f_std': meso_solution.std()}
meso_x = pd.DataFrame(meso_x)
meso_x.to_csv('result/modifications/Himmelblau_result.csv')

# In[]
data = {}

pso_solution = []
ga_solution = []
de_solution = []
eso_solution = []
for i in range(no_of_runs):
    
    pso = PSO(func=f_Himmelblau, dim=n_dim, pop=size_pop, max_iter=max_iter, lb=lb, ub=ub, w=0.8, c1=0.5, c2=0.5)
    pso.run()
    data['Himmelblau_solution_pso_{}'.format(i)] = pso.gbest_y_hist
    pso_solution.append(pso.gbest_y_hist[-1])
    print('pso: ', pso.gbest_y_hist[-1])
    
    de = DE(func=f_Himmelblau, n_dim=n_dim, size_pop=size_pop, max_iter=max_iter, lb=lb, ub=ub)
    de.run()
    data['Himmelblau_solution_de_{}'.format(i)] = de.generation_best_Y
    de_solution.append(de.generation_best_Y[-1])
    print('de: ', de.generation_best_Y[-1])
    
    
    ga = GA(func=f_Himmelblau, n_dim=n_dim, size_pop=size_pop, max_iter=max_iter, prob_mut=0.001, lb=lb, ub=ub, precision=1e-7)
    ga.run()
    data['Himmelblau_solution_ga_{}'.format(i)] = ga.generation_best_Y
    ga_solution.append(ga.generation_best_Y[-1])
    print('ga: ', ga.generation_best_Y[-1])

    eso = ESO(f_Himmelblau, n_dim, size_pop, max_iter, lb, ub)
    eso.run()
    data['Himmelblau_solution_eso_{}'.format(i)] = eso.y_history
    eso_solution.append(eso.y_global_best)
    

data = pd.DataFrame(data)
data.to_csv('result/modifications/Himmelblau_solution_others.csv')