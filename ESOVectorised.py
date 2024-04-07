# -*- coding: utf-8 -*-
import numpy as np
import math


# In[]
class ESO:
    def __init__(self, func, n_dim, population_size, max_iter, lb, ub, constraint_ueq=None):
        
        self.func     = func
        
        self.n_dim    = n_dim
        self.population_size = population_size
        self.matrix_dim = (self.population_size, self.n_dim)
        self.max_iter = max_iter
        self.lb   = lb
        self.ub   = ub
        
        # adam's learning rate of weight estimate
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.m = np.zeros(self.matrix_dim)
        self.v = np.zeros(self.matrix_dim)
        self.w = np.random.uniform(-1, 1, size=self.matrix_dim)
        # Empty matrix with same pop_size * n dimenensions and garbage values.
        self.g = np.empty_like(self.w)
        
        # location, fitness, and estimate fitness
        # Initialise x to random values between lb to ub (-100 to 100 in this case)
        self.x = np.random.uniform(0, 1, size=self.matrix_dim) * (self.ub-self.lb) + self.lb
        self.y = np.empty(self.population_size)
        self.y_hat = self.y.copy()
        
        # best fitness history and estimate error history
        self.y_history = []
        self.error_history = []
        
        # individual best location, gradient direction, and fitness 
        self.x_hist_best = self.x.copy()
        self.g_hist_best = np.empty_like(self.x)
        self.y_hist_best = np.ones(population_size)*np.inf
        
        # group best location, gradient direction, and fitness 
        self.x_global_best = self.x[0].copy()
        self.g_global_best = np.zeros(self.n_dim)
        self.y_global_best = func(self.x[0].reshape(1, self.n_dim))
        
        self.hop = self.ub - self.lb
        
    def callFunc(self, x):
        y = self.func(x)
        if (len(y.shape) == 0):
            y = np.tile(y, self.population_size).T
        return y
    
    def clipToMeetBounds(self,x):
        return np.clip(x, self.lb, self.ub)
    
    

    def refill(self,V):
        V = V.reshape(len(V), 1)
        V = np.tile(V, self.n_dim)
        return V
    
    def gradientEstimate(self, g_temp):
        
        # Indivual direction
        p_d = self.x_hist_best - self.x
        p_d_sum = p_d.sum(axis=1)
        p_d_sum = self.refill(p_d_sum)
        f_p_bias = self.y_hist_best - self.y
        f_p_bias = self.refill(f_p_bias)
        p_d *= f_p_bias
        p_d /= (p_d_sum+np.spacing(1))*(p_d_sum+np.spacing(1))
        
        d_p = p_d + self.g_hist_best
        
        # Group direction
        c_d = self.x_global_best - self.x
        c_d_sum = c_d.sum(axis=1)
        c_d_sum = self.refill(c_d_sum)
        f_c_bias = self.y_global_best - self.y
        f_c_bias = self.refill(f_c_bias)
        c_d *= f_c_bias
        c_d /= (c_d_sum+np.spacing(1))*(p_d_sum+np.spacing(1))
        
        d_g = c_d + self.g_global_best
        
        # Advice
        r1 = np.random.random(self.population_size)
        r1 = self.refill(r1)
        
        r2 = np.random.random(self.population_size)
        r2 = self.refill(r2)
        
        r3 = np.random.random(self.population_size)
        r3 = self.refill(r3)
        
        self.g = r1 * g_temp + r2 * d_p + r3 * d_g
        g_sum = self.g.sum(axis=1)
        g_sum = self.refill(g_sum)
        self.g /= (g_sum+np.spacing(1))
    
    def weightUpdate(self):
        # Update weight
        # momentum
        self.m = self.beta1*self.m+(1-self.beta1)*self.g
        # variance
        self.v = self.beta2*self.v+(1-self.beta2)*self.g**2
        self.w = self.w - self.m/(np.sqrt(self.v)+np.spacing(1))
    
    def updateSurface(self):
        self.y = self.callFunc(self.x)
        self.y_hat = np.sum(self.w*self.x, axis=1)
        self.error_history.append(np.abs(self.y-self.y_hat).min())
        p = self.y_hat-self.y
        p = self.refill(p)
        g_temp = p*self.x
        
        mask = self.y < self.y_hist_best
        self.y_hist_best = np.where(mask, self.y, self.y_hist_best)
        
        mask = self.refill(mask)
        self.x_hist_best = np.where(mask, self.x, self.x_hist_best)
        self.g_hist_best = np.where(mask, g_temp, self.g_hist_best)
        
        g_hist_sum = self.refill(np.sqrt((self.g_hist_best**2).sum(axis=1)))
        self.g_hist_best /= (g_hist_sum + np.spacing(1))
        
        # Data Insecure
        if self.y.min() < self.y_global_best:
            self.y_global_best = self.y.min()
            self.x_global_best = self.x[self.y.argmin(), :]
            self.g_global_best = g_temp[self.y.argmin(), :]
            self.g_global_best /= np.sqrt(np.sum(self.g_global_best**2)) 
        
        self.gradientEstimate(g_temp)
        
        self.weightUpdate()
        
    

    def randomSearch(self):
        # Random search
        r = np.random.uniform(-np.pi / 2, np.pi / 2, size=self.matrix_dim)
        x_b = self.x + np.tan(r) * self.hop/( 1 + self.current_iter) *0.5
        
        x_b = self.clipToMeetBounds(x_b)

        y_b = self.callFunc(x_b)
        
        # Random step search
        d = self.x_hist_best - self.x
        d_g = self.x_global_best - self.x
        #r = np.random.uniform(-np.pi / 2, np.pi / 2, size=(population_size, n_dim))
        r = np.random.uniform(0, 0.5, size=self.matrix_dim)
        r2 = np.random.uniform(0, 0.5, size=self.matrix_dim)
        x_c = (1-r-r2) * self.x + r * d + r2 * d_g
        x_c = self.clipToMeetBounds(x_c)
        
        y_c = self.callFunc(x_c)

        return x_c, y_c, x_b, y_b


    def adviceSearch(self):
        x_a = self.x + np.exp(-self.current_iter/(0.1*self.max_iter)) * 0.1 * self.hop * self.g
        x_a = self.clipToMeetBounds(x_a)

        
        y_a = self.callFunc(x_a)
        return x_a, y_a

    
    def run(self):
        for self.current_iter in range(self.max_iter):
            self.updateSurface()
            x_c, y_c, x_b, y_b = self.randomSearch()
            x_a, y_a = self.adviceSearch()
            
            # Comparison
            x_i = np.empty_like(self.x)
            y_i = np.empty_like(self.y)
            x_summary = np.array([x_c, x_b, x_a])
            y_summary = np.column_stack((y_c, y_b, y_a))
            y_summary[y_summary == np.nan] = np.inf
            i_ind = y_summary.argmin(axis=1)
            for i in range(self.population_size):
                y_i[i] = y_summary[i, i_ind[i]]
                x_i[i,:] = x_summary[i_ind[i]][i]

            
            # Update location
            mask = y_i < self.y
            self.y = np.where(mask, y_i, self.y)
            
            mask = self.refill(mask)
            self.x = np.where(mask, x_i, self.x)
            
            
            mask = y_i < self.y_hist_best
            self.y_hist_best = np.where(mask, y_i, self.y_hist_best)
            
            mask = self.refill(mask)
            self.x_hist_best = np.where(mask, x_i, self.x_hist_best)
            
            if y_i.min() < self.y_global_best:
                self.y_global_best = y_i[y_i.argmin()]
                self.x_global_best = x_i[y_i.argmin(), :]
            else:
                ran = np.random.random(self.population_size)
                ran = self.refill(ran)
                ran[mask] = 1
                mask = ran < 0.3
                self.x = np.where(mask, x_i, self.x)
                self.y = np.where(mask[:, 0], y_i, self.y)
            

            

            self.y_history.append(self.y_global_best.copy())


