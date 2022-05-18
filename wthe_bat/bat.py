import numpy as np
from numpy.random import random as rand

class BatAlgorithm():
    def __init__(self, obj_func, N_pop, N_gen, A, r, Qmin, Qmax, dimension, lower, upper):
        self.obj_func = obj_func    # objective function
        self.N_pop = N_pop  # population size
        self.N_gen = N_gen  # generation size
        self.A = A  # loudness
        self.r = r  # pulse rate
        self.Qmin = Qmin    # frequency minimum
        self.Qmax = Qmax    # frequency maximum
        self.dimension = dimension  # dimension
        self.lower = lower
        self.upper = upper
        self.LB = lower * np.zeros((1, self.dimension)) # lower bound
        self.UB = upper * np.zeros((1, self.dimension)) # upper bound
        self.Q = np.zeros((self.N_pop, 1))  # frequency of the swarm
        self.V = np.zeros((self.N_pop, self.dimension))  # velocity of the swarm
        self.solution = np.zeros((self.N_pop, self.dimension))  # population of solution
        self.min_fitness = 0.0  # minimum fitness
        self.fitness = np.zeros((self.N_pop, 1))    # fitness
        self.best = np.zeros((1, self.dimension))   # best solution
    
    def best_bat(self):
        self.min_fitness = min(self.fitness)
        self.best = self.solution[list(self.fitness).index(self.min_fitness)]

    def init_bat(self):
        self.LB = self.lower * np.ones((1, self.dimension))
        self.UB = self.upper * np.ones((1, self.dimension))
        self.Q = np.zeros((self.N_pop, 1))
        self.V = np.zeros((self.N_pop, self.dimension))
        # prepare initial solution
        for i in range(self.N_pop):
            self.solution[i] = np.random.uniform(self.lower, self.upper, (1, self.dimension))
            self.fitness[i] = self.obj_func(self.solution[i])
        # find the initial optimal solution
        self.best_bat()
    
    def simplebounds(self, val, LB, UB):
        # print(val)
        # print(LB)
        for i in range(self.dimension):
            if val[i] < LB[i]: val[i] = LB[i]
            if val[i] > UB[i]: val[i] = UB[i]
        return val
    
    def move_bat(self):
        S = np.zeros((self.N_pop, self.dimension))  # the location of the bats
        self.init_bat()

        for t in range(self.N_gen):
            for i in range(self.N_pop):
                self.Q[i] = np.random.uniform(self.Qmin, self.Qmax)
                self.V[i] += (self.solution[i] - self.best) * self.Q[i]
                S[i] = self.solution[i] + self.V[i]
                # apply simple boundary
                S[i] = self.simplebounds(self.solution[i], self.LB[0], self.UB[0])
                # pulse rate
                if np.random.random() > self.r:
                    # generate local solution around the selected best solution
                    S[i] = self.best + 0.001 * np.random.randn(1, self.dimension)
                    S[i] = self.simplebounds(S[i], self.LB[0], self.UB[0])
                # evaluate new solutions
                Fnew = self.obj_func(S[i])
                # if new solution improves and not too loud, update solution
                if Fnew <= self.fitness[i] and rand() < self.A:
                    self.solution[i] = S[i]
                    self.fitness[i] = Fnew
                # update the current best solution
                if Fnew <= self.min_fitness:
                    self.best = S[i]
                    self.min_fitness = Fnew
            
        print("Best =", self.best, "\nfmin =", self.min_fitness)
        return self.best

def Fun(sol):
    val = 0.0
    for s in sol:
        val += s * s
    return val

# For reproducive results
#random.seed(5)

if __name__ == "__main__":
    BA = BatAlgorithm(Fun, 40, 1000, 0.5, 0.5, 0, 2, 10, -10, 10)
    print(BA.move_bat())