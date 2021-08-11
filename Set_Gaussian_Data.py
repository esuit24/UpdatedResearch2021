#!/usr/bin/env python
# coding: utf-8



#imports
import math
import numpy as np
from random import uniform


    #each sample has a bunch of different x values to represent the given Gaussian
    #each gaussian is defined by the mean and the standard deviation so choosing discrete x values to identify
    #the function with can pinpoint different areas on the graph to choose a mean value


#evaluate gaussian function result given x, mean, and sigma: outputs array of gaussian results that corresponds to each x value in input array
def gaussian(num_x_vals, mean, sigma = 1):
    samples = np.zeros((num_x_vals,))
    c = 1/(sigma * math.sqrt(2*math.pi))
    x_val = -1
    increment = 2/(num_x_vals-1)
    index = 0
    while(x_val <= 1):
        exp = -((x_val-mean)**2) / (2*sigma**2)
        samples[index] = c*(math.e**exp)
        index += 1
        x_val += increment
    return samples
#create training set with num_samples samples (# rows) and num_x_vals number of discrete x values (columns), returns 2D array where each row
#is a different sample of disctete x values
def create_training_set(num_samples, num_x_vals = 21):
    training_data = np.zeros((num_samples, num_x_vals))
    means = np.zeros((num_samples,))
    sigmas = np.zeros((num_samples,))
    sample = 0
    while (sample < len(training_data)):
        random_mean = uniform(-0.25, 0.25) #mean domain is -0.25, 0.25
        random_sigma = uniform(-0.25, 0.25) #sd domain is -0.25, 0.25
        means[sample] = random_mean
        sigmas[sample] = random_sigma
        training_data[sample] = gaussian(num_x_vals, random_mean, random_sigma)
        sample += 1
    return training_data, means, sigmas
