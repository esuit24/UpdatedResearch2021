#imports
import math
import numpy as np
from random import uniform
from scipy.integrate import quad
from scipy.special import gamma

#evauate normal distribution function output based on location, scale, shape, and x values: outputs array with values that correspond to each
#x input value
def function(location, scale, shape, num_x_vals): #need a range of x values to choose from
    curr_x = -5
    index = 0
    vals = np.zeros((num_x_vals,))
    increment = 10.0/(num_x_vals-1)
    while(curr_x <= 5):
        c = shape/(2*scale*gamma(curr_x)*(1/shape))
        exp_val = (abs(curr_x-location)/scale)**shape
        ex = -exp_val
        vals[index] = c* math.e ** (ex)
        curr_x += increment
        index += 1
    return vals

#creates a training set with num_samples samples and num_x_vals discrete x values, each row is a new sample of discrete x values
#each column is a different disctete x value that defines the function
def create_training_set(num_samples, num_x_vals = 21):
    training_data = np.zeros((num_samples, num_x_vals))
    mu_vals = np.zeros((num_samples,))
    alpha_vals = np.zeros((num_samples,))
    beta_vals = np.zeros((num_samples,))
    for i in range(num_samples):
        #initializing labels
        random_mu = uniform(-0.25, 0.25) #mu domain is -0.25, 0.25
        mu_vals[i] = random_mu
        random_alpha = uniform(1,2) #alpha domain is 1, 2
        alpha_vals[i] = random_alpha
        random_beta = uniform(1,5) #beta domain is 1,5
        beta_vals[i] = random_beta
        training_data[i] = function(random_mu, random_alpha, random_beta, num_x_vals)
    scale(mu_vals)
    scale(alpha_vals)
    scale(beta_vals)
    return training_data, mu_vals, alpha_vals, beta_vals

#scale values so their RMS are all order one (different domains so scale so that each contribute equally to error calculation)
def scale(vals):
    sum = 0
    for i in range(len(vals)):
        sum += vals[i] * vals[i]
    scale_factor = math.sqrt((1/len(vals)) * sum)
    for j in range(len(vals)):
        vals[j] = vals[j] / scale_factor
    return vals
