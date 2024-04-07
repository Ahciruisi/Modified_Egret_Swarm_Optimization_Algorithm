# -*- coding: utf-8 -*-
# In[]
import numpy as np
import math


# In[]
class ESOModified:
    def __init__(self, func, n_dim, population_size, max_iter, lb, ub, constraint_ueq=None):     
        self.func     = func
        
        # Size of the variable vector.
        # For a function func(x), x is not a single number; it is a set of variables x1, x2, ..., xn.
        self.n_dim    = n_dim
        # Size of the population/Number of agents/Number of egret swarms.
        # Note that a swarm = a collection of egret squads here.
        # In most NIOs, we usually work with multiple values of x, or multiple agents in a single run
        # for best results.
        self.population_size = population_size
        self.matrix_dim = (self.population_size, self.n_dim)
        self.max_iter = max_iter
        self.lb   = lb
        self.ub   = ub
        
        # adam's learning rate of weight estimate
        self.beta1 = 0.9
        self.beta2 = 0.99
        # Momentum; used to calculate directional correction for egret A
        self.m = np.zeros(self.matrix_dim)
        # Variance; used to calculate directional correction for egret A
        self.v = np.zeros(self.matrix_dim)
        # Weights; used to calculate the predicted value of f(x)
        self.w = np.random.uniform(-1, 1, size=self.matrix_dim)
        # Gradient; empty matrix with same dimensions (population_size * n dim) and garbage values.
        self.g = np.empty_like(self.w)
        
        # location, fitness (f(x) or y), and estimate fitness (y_hat)
        # Initialise all elements in x to random values between lb to ub (-100 to 100 in this case)
        self.x = np.random.uniform(0, 1, size=self.matrix_dim) * (self.ub-self.lb) + self.lb
        # fitness/objective function: empty vector of length population_size and garbage values.
        # The length of this vector is population_size because it holds the values of f(x) for each swarm.
        self.y = np.empty(self.population_size)
        # estimated fitness/objective function: empty vector of length population_size and garbage values.
        self.y_hat = self.y.copy()
        
        # best fitness history and estimate error history
        self.y_history = []
        self.error_history = []
        
        # best location, gradient direction, and fitness of each individual swarm/agent/population.
        # All these matrices are the same dimensions as x, w, etc i.e. (population_size x n_dim)
        self.x_hist_best = self.x.copy()
        self.g_hist_best = np.empty_like(self.x)
        # Vector of length population_size
        self.y_hist_best = np.ones(population_size)*np.inf
        
        # best location, gradient direction, and fitness across the whole population/all swarms/all agents.
        # Since these variables just hold the global best values i.e. the best value among population_size values,
        # these variables are arrays of length n_dim.
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
        """Turns a vector V of length l into a matrix of size (l, n_dim) by repeating the vector n_dim times."""
        V = V.reshape(len(V), 1)
        V = np.tile(V, self.n_dim)
        return V
    
    def gradientEstimate(self, g_temp):
        # Individual/Personal direction of each swarm.
        d_p = self.x_hist_best - self.x
        # Sum across all d_p_1, d_p_2, ...., d_p_n
        # Vector of length "population_size"
        d_p_sum = d_p.sum(axis=1)
        # Matrix of size (population_size, n_dim)
        d_p_sum = self.refill(d_p_sum)
        f_p_bias = self.y_hist_best - self.y
        f_p_bias = self.refill(f_p_bias)
        d_p *= f_p_bias
        d_p /= (d_p_sum+np.spacing(1))*(d_p_sum+np.spacing(1))
        
        d_p = d_p + self.g_hist_best
        
        # Group direction of the population as a whole.
        d_g = self.x_global_best - self.x
        d_g_sum = d_g.sum(axis=1)
        d_g_sum = self.refill(d_g_sum)
        f_c_bias = self.y_global_best - self.y
        f_c_bias = self.refill(f_c_bias)
        d_g *= f_c_bias
        d_g /= (d_g_sum+np.spacing(1))*(d_p_sum+np.spacing(1))
        
        d_g = d_g + self.g_global_best
        
        # Advice
        r1 = np.random.random(self.population_size)
        r1 = self.refill(r1)
        
        r_h = np.random.random(self.population_size)
        r_h = self.refill(r_h)
        
        r_g = np.random.random(self.population_size)
        r_g = self.refill(r_g)
        
        self.g = r1 * g_temp + r_h * d_p + r_g * d_g
        g_sum = self.g.sum(axis=1)
        g_sum = self.refill(g_sum)
        self.g /= (g_sum+np.spacing(1))
    
    def weightUpdate(self):
        # Update weights and related parameters using Adam optimisation.
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
        
        # Essentially, this updates y_hist_best with values from y wherever
        # the value in y is smaller.
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
        

    def levySearch(self):
        beta = 1.5
        sigma_v = 1
        sigma_u = ( ( math.gamma(1+beta)*np.sin(np.pi*beta/2) ) / ( beta*math.gamma((1+beta)/2)*2**((beta-1)/2) ) )**(1/beta)
        
        u = np.random.normal(0, sigma_u**2, size=self.matrix_dim)
        v = np.random.normal(0, sigma_v**2, size=self.matrix_dim)
        s = u/(np.abs(v)**(1/beta))
        # Adjust the value of s to be in between lb and ub.
        x = s * (self.ub - self.lb) / self.lb + self.lb
        
        return x


    def randomSearch(self):
        decay = 1 / (1 + self.current_iter)

        # Random search: Egret B's random search.
        # Note that with increasing number of iterations, the value of the coefficient "decay"
        # decreases, which causes less drastic updates to x_b. But also, this reduces the chance
        # of egret B "escaping" off even if r = -pi/2 or pi/2 (at which tan(r) becomes infinity). 
        r = np.random.uniform(-np.pi / 2, np.pi / 2, size=self.matrix_dim)
        x_b = self.x + np.tan(r) * self.hop * decay * 0.5
        
        x_b = self.clipToMeetBounds(x_b)
        y_b = self.callFunc(x_b)
        
        # Levy search: Egret D
        # Here's the new addition to the algorithm:
        # We added one more source of randomness in the search using a modified version of Levy's flight.
        # We use Levy's flight to calculate the next *proposed* position, but instead of directly updating
        # x_d to proposed_x, we first multiply it by a decay coefficient. This way, we avoid drastic updates
        # to the position when we're nearing the end of the search.
        # Note that the decay is also used for Egret B's random search.
        proposed_x = self.levySearch()
        delta = (proposed_x - self.x)
        x_d = self.x + delta * self.hop * decay * 0.5

        x_d = self.clipToMeetBounds(x_d)
        y_d = self.callFunc(x_d)

        # Random step search: Egret C's aggressive encircling mechanism
        # Updates egret C's position using the value of the directional correction of a swarm/agent/population,
        # as well as the directional correction of the best swarm/agent/population. The weights given to each
        # direction are randomly sampled from a uniform distribution. Since we use a *uniform* distribution here,
        # this is an aggressive random walk.
        d = self.x_hist_best - self.x
        d_g = self.x_global_best - self.x
        #r = np.random.uniform(-np.pi / 2, np.pi / 2, size=(population_size, n_dim))
        r = np.random.uniform(0, 0.5, size=self.matrix_dim)
        r2 = np.random.uniform(0, 0.5, size=self.matrix_dim)
        x_c = (1-r-r2) * self.x + r * d + r2 * d_g
        
        x_c = self.clipToMeetBounds(x_c)
        y_c = self.callFunc(x_c)

        return x_d, y_d, x_c, y_c, x_b, y_b


    def adviceSearch(self):
        x_a = self.x + np.exp(-self.current_iter/(0.1*self.max_iter)) * 0.1 * self.hop * self.g

        x_a = self.clipToMeetBounds(x_a)        
        y_a = self.callFunc(x_a)
        return x_a, y_a

    
    def run(self):
        for self.current_iter in range(self.max_iter):
            # Compute directional corrections and such for egret A.
            self.updateSurface()
            # Compute suggested values of x from egrets B, C, D.
            x_d, y_d, x_c, y_c, x_b, y_b = self.randomSearch()
            # Compute suggested value of x from egrets A.
            x_a, y_a = self.adviceSearch()
            
            # Comparison
            x_i = np.empty_like(self.x)
            y_i = np.empty_like(self.y)
            x_summary = np.array([x_d, x_c, x_b, x_a])
            # Matrix of size (population_size x 4), since each y is stacked side by side.
            # For example, if y_d = [4, 2, 5], y_c = [0, 5, 4], y_b = [7, 8, 9], y_a = [1, 1, 1]
            # y_summary would be:
            # [[4, 0, 7, 1],
            #  [2, 5, 8, 1],
            #  [5, 4, 9, 1]]
            y_summary = np.column_stack((y_d, y_c, y_b, y_a))
            # Replace any invalid values with infinity.
            y_summary[y_summary == np.nan] = np.inf
            # Vector of length population_size.
            # Each row stores the column number at which you can find the lowest y.
            # For example, the argmin of the above y_summary example will give:
            # [1,
            #  3,
            #  3]
            best_egrets_for_each_swarm = y_summary.argmin(axis=1)
            # For each swarm, see which egret gave it the best result and store it.
            for swarm_i in range(self.population_size):
                y_i[swarm_i] = y_summary[swarm_i, best_egrets_for_each_swarm[swarm_i]]
                x_i[swarm_i,:] = x_summary[best_egrets_for_each_swarm[swarm_i]][swarm_i]

            
            # Update location
            mask = y_i < self.y
            self.y = np.where(mask, y_i, self.y)
            
            mask = self.refill(mask)
            self.x = np.where(mask, x_i, self.x)
            
            
            mask = y_i < self.y_hist_best
            self.y_hist_best = np.where(mask, y_i, self.y_hist_best)
            
            mask = self.refill(mask)
            self.x_hist_best = np.where(mask, x_i, self.x_hist_best)
            
            # If the lowest y in this iteration is lower than our global best, update the global best.
            if y_i.min() < self.y_global_best:
                self.y_global_best = y_i[y_i.argmin()]
                self.x_global_best = x_i[y_i.argmin(), :]
            else:
                probability_of_updating = np.random.random(self.population_size)
                probability_of_updating = self.refill(probability_of_updating)
                # mask has the indices where the current iteration's y is better than the personal bests.
                # Update the probability of updating to 1 wherever the current y is better.
                probability_of_updating[mask] = 1
                # Wherever the probablity of updating is >= 0.3, update it.
                mask = probability_of_updating < 0.3
                self.x = np.where(mask, x_i, self.x)
                self.y = np.where(mask[:, 0], y_i, self.y)

            self.y_history.append(self.y_global_best.copy())


