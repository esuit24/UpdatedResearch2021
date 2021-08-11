#imports
import numpy as np
from scipy.integrate import quad
from random import uniform

#evaluate function based on x value and coefficient values, returns single value that represents result of equation a + bx + cx^2 ....
def func(x, cfs):
    res = 0
    for i in range(len(cfs)):
        res += cfs[i] * x ** i
    return res
#evalutates the integrand to be integrated
def integrand(x, power, cfs):
    return x**power * func(x, cfs)
#evaluates the moment- the integral of the above equation
def evaluate_moment(power, cfs):
    res = quad(integrand, -1, 1, args = (power, cfs))
    return  res[0] #domain = (-1,1)
#randomly generate coefficients
def gen_coefs(num_vals, lower, upper): #num_vals is the number of coefficients
    coefs = np.zeros((num_vals,))
    for i in range(len(coefs)):
        random_val = uniform(lower,upper) #coefficient domain = (lower,upper)
        coefs[i] = random_val
    return coefs
#generate training data and outputs 2D array with each row being a new sample of moment data and each column representing a different moment
#num_cs = number of coefficients to be generated
#outputs training dataset with answers 
def generate_training_data(num_samples, num_moments, num_cs):
    samples = np.zeros((num_samples, num_moments))
    labels = np.zeros((num_samples, num_cs))
    #generate corresponding coefs for labels
    for i in range(num_samples):
        cs = gen_coefs(num_cs, -20, 20) #each new sample has a different set of coefficients
        labels[i] = cs
        for j in range(num_moments):
            samples[i][j] = evaluate_moment(j, cs)
    return samples, labels
