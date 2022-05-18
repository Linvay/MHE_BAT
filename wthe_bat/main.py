from tkinter.tix import COLUMN, ROW
import numpy as np
import cv2
from imWeightedThresholdedheq import imWTHeq
import math
import skimage.measure
import numpy as np
from numpy.random import random as rand
from matplotlib import pyplot as plt

class wthe_bat():
    def __init__(self, og_img, N_pop, N_gen, A, r, Qmin, Qmax, alpha=0.9, gamma=0.9, Vmin=-0.2, Vmax=0.2, lower=0.1, upper=1, dimension=2):
        self.og_img = og_img
        self.og_img_entropy = skimage.measure.shannon_entropy(self.og_img)
        self.N_pop = N_pop  # population size
        self.N_gen = N_gen  # generation size
        self.A0 = A  # loudness
        self.r0 = r  # pulse rate
        self.Qmin = Qmin    # frequency minimum
        self.Qmax = Qmax    # frequency maximum
        self.alpha = alpha
        self.gamma = gamma
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.dimension = dimension  # dimension
        self.lower = lower
        self.upper = upper
        self.A = np.zeros((self.N_pop, 1))
        self.r = np.zeros((self.N_pop, 1))
        self.LB = np.zeros((1, self.dimension)) # lower bound
        self.UB = np.zeros((1, self.dimension)) # upper bound
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
        self.A = self.A0 * np.ones((self.N_pop, 1))
        self.r = self.r0 * np.ones((self.N_pop, 1))
        self.LB = self.lower * np.ones((1, self.dimension))
        self.UB = self.upper * np.ones((1, self.dimension))
        self.Q = np.zeros((self.N_pop, 1))
        self.V = np.zeros((self.N_pop, self.dimension))
        # prepare initial solution
        for i in range(self.N_pop):
            self.solution[i] = np.random.uniform(self.lower, self.upper, (1, self.dimension))
            self.fitness[i] = self.img_entropy_dif(self.solution[i])
        # find the initial optimal solution
        self.best_bat()
    
    def simplebounds(self, val, LB, UB):
        for i in range(self.dimension):
            if val[i] < LB[i]: val[i] = LB[i]
            if val[i] > UB[i]: val[i] = UB[i]
        return val
    
    def search_optimal(self):
        S = np.zeros((self.N_pop, self.dimension))  # the location of the bats
        self.init_bat()

        for t in range(self.N_gen):
            for i in range(self.N_pop):
                self.Q[i] = np.random.uniform(self.Qmin, self.Qmax)
                self.V[i] += (self.solution[i] - self.best) * self.Q[i]
                S[i] = self.solution[i] + self.V[i]
                # apply simple boundary
                S[i] = self.simplebounds(S[i], self.LB[0], self.UB[0])
                self.V[i] = self.simplebounds(self.V[i], [self.Vmin] * 2, [self.Vmax] * 2)
                # pulse rate
                if np.random.random() > self.r[i]:
                    # generate local solution around the selected best solution
                    S[i] = self.best + 0.001 * np.random.randn(1, self.dimension)
                    S[i] = self.simplebounds(S[i], self.LB[0], self.UB[0])
                # evaluate new solutions
                Fnew = self.img_entropy_dif(S[i])
                # if new solution improves and not too loud, update solution
                if Fnew <= self.fitness[i] and rand() < np.average(self.A):
                    self.solution[i] = S[i]
                    self.fitness[i] = Fnew
                    # update Ai and ri
                    self.A[i] *= self.alpha
                    self.r[i] = self.r0 * (1 - math.exp(-self.gamma * t))
                    # print(1 - math.exp(-self.gamma * t))
                    # print("A[" + str(i) + "] =", self.A[i])
                    # print("r[" + str(i) + "] =", self.r[i])
                # update the current best solution
                if Fnew <= self.min_fitness:
                    self.best = S[i]
                    self.min_fitness = Fnew
            print(t)
            
        print("Best =", self.best, "\nfmin =", self.min_fitness)
        heq_img, Wout = imWTHeq(self.og_img, r=self.best[0], v=self.best[1])
        return heq_img
    
    def img_entropy_dif(self, sol):
        heq_img, Wout = imWTHeq(self.og_img, r=sol[0], v=sol[1])
        heq_img_entropy = skimage.measure.shannon_entropy(heq_img)
        return abs(heq_img_entropy - self.og_img_entropy)

if __name__ == "__main__":
    test_img_dir = "test_img/"
    result_img_dir = "result_img/"
    og_img_name = "einstein"
    og_img_fname = og_img_name + ".jpg"
    print(og_img_fname)
    og_img = cv2.imread(test_img_dir + og_img_fname, cv2.IMREAD_GRAYSCALE)

    MHE = wthe_bat(og_img, 50, 100, 1, 0.001, 0, 2)
    mhe_img = MHE.search_optimal()

    he_img = cv2.equalizeHist(og_img)

    # create figure to display multiple image
    display = plt.figure(figsize=(10, 7))
    # setting values of rows and columns
    row = 2
    column = 3

    display.add_subplot(row, column, 1)
    plt.imshow(og_img, cmap='gray')
    plt.axis('off')
    plt.title("Original")

    display.add_subplot(row, column, 2)
    plt.imshow(he_img, cmap='gray')
    plt.axis('off')
    plt.title("HE")

    display.add_subplot(row, column, 3)
    plt.imshow(mhe_img, cmap='gray')
    plt.axis('off')
    plt.title("MHE")

    display.add_subplot(row, column, 4)
    plt.xlabel("Value")
    plt.ylabel("Pixel Frequency")
    plt.hist(og_img.ravel(), 256, [0, 255])

    display.add_subplot(row, column, 5)
    plt.xlabel("Value")
    plt.ylabel("Pixel Frequency")
    plt.hist(he_img.ravel(), 256, [0, 255])

    display.add_subplot(row, column, 6)
    plt.xlabel("Value")
    plt.ylabel("Pixel Frequency")
    plt.hist(mhe_img.ravel(), 256, [0, 255])

    plt.savefig(result_img_dir + og_img_name)
    plt.show()

    cv2.imwrite(result_img_dir + "mhe_" + og_img_fname, mhe_img)
    cv2.imwrite(result_img_dir + "he_" + og_img_fname, he_img)