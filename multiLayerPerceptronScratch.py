#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:54:40 2023

Make a Multi-Layer Perceptron Model for classification from scratch in Python.

@author: paulmason
"""

import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

#Find a small float to avoid division by 0
epsilon = np.finfo(float).eps 

#Sigmoid function & differentiation - used in output layer
def sigmoid(z):
    #clip limits value to be between -500 & 500 to avoid overflow
    return 1 / (1 + np.exp(-z.clip(-500, 500)))

#Derivative of the sigmoid function
def dsigmoid(z):
    s = sigmoid(z)
    return 2 * s * (1 - s)

#ReLU function & differentiation - used in hidden layers
def relu(z):
    return np.maximum(0, z)

#Derivative of ReLU function
def drelu(z):
    return (z > 0).astype(float)

#Loss function L(y, yhat) & its differentiation
def cross_entropy(y, yhat):
    '''Binary cross entropy function
        L = -y log yhat - (1 - y) log (1 - yhat)
        
        Args:
            y, yhat (np.array): 1xn matrices which n are the # of data instances
        Returns:
            average cross entropy value of shape 1x1, averaging over the n instances
    '''
    return -(y.T @ np.log(yhat.clip(epsilon)) + (1 - y.T) @ np.log((1 - yhat).clip(epsilon))) / y.shape[1]

def d_cross_entropy(y, yhat):
    # dL / dyhat
    return - np.divide(y, yhat.clip(epsilon)) + np.divide(1 - y, (1 - yhat).clip(epsilon))


#Create a class to encapsulate entire MLP model
class mlp:
    #Multilayer perceptron using numpy
    def __init__(self, layersizes, activations, derivatives, lossderiv):
        #Remember config, then initialize array to hold NN parameters without init
        #Hold NN config
        self.layersizes = layersizes
        self.activations = activations
        self.derivatives = derivatives
        self.lossderiv = lossderiv
        
        assert len(self.layersizes) - 1 == len(self.activations), \
            "Number of layers and number of activation functions does not match."
            
        assert len(self.activations) == len(self.derivatives), \
            "Number of activation functions and number of derivatives does not match."
            
        assert all(isinstance(n, int) and n >= 1 for n in layersizes), \
            "Only positive integral number of perceptons is allowed in each layer."
        
        #Parameters, each is a 2D numpy array
        L = len(self.layersizes)
        #Forward pass
        self.z = [None] * L
        self.W = [None] * L
        self.b = [None] * L
        self.a = [None] * L
        #Gradients for back-propagation
        self.dz = [None] * L
        self.dW = [None] * L
        self.db = [None] * L
        self.da = [None] * L
    
    #Initialize value of weight matrices and bias vectors with small random numbers
    def initialize(self, seed = 42):
        np.random.seed(seed)
        sigma = 0.1
        for l, (insize, outsize) in enumerate(zip(self.layersizes, self.layersizes[1:]), 1):
            self.W[l] = np.random.randn(insize, outsize) * sigma
            self.b[l] = np.random.randn(1, outsize) * sigma
     
    #Transform network's input x to output by loop in this method        
    def forward(self, x):
        self.a[0] = x
        for l, func in enumerate(self.activations, 1):
            #z = W a + b, with 'a' as output from previous layer
            #'W' is of size rxs and 'a' the size sxn with n the # of data instances, 'z' the size rxn
            #'b' is rx1 and broadcast to each column of 'z'
            self.z[l] = (self.a[l-1] @ self.W[l]) + self.b[l]
            #a = g(z), with 'a' as output of this layer, of size rxn
            self.a[l] = func(self.z[l])
        return self.a[-1]
    
    #Run backpropagation to compute the gradient of the weight & bias of each layer
    def backward(self, y, yhat):
        assert y.shape[1] == self.layersizes[-1], "Output size doesn't match network output size"
        assert y.shape == yhat.shape, "Output size doesn't match reference"
        #First 'da', at the output
        self.da[-1] = self.lossderiv(y, yhat)
        for l, func in reversed(list(enumerate(self.derivatives, 1))):
            #Compute the differentials at this layer
            self.dz[l] = self.da[l] * func(self.z[l])
            self.dW[l] = self.a[l - 1].T @ self.dz[l]
            self.db[l] = np.mean(self.dz[l], axis = 0, keepdims = True)
            self.da[l - 1] = self.dz[l] @ self.W[l].T

    #Apply gradients found by back-propagation to the parameters b and W
    def update(self, eta):
        for l in range(1, len(self.W)):
            self.W[l] -= eta * self.dW[l]
            self.b[l] -= eta * self.db[l]
            
            
#Make data: 2 circles on x-y plane as a classification problem
x, y = make_circles(n_samples = 1000, factor = 0.5, noise = 0.1)
#Model expects a 2D array
y = y.reshape(-1, 1)    

#Build MLP model with input layer, 2 hidden layers and an output layer
model = mlp(layersizes = [2, 4, 3, 1],
            activations = [relu, relu, sigmoid],
            derivatives = [drelu, drelu, dsigmoid],
            lossderiv = d_cross_entropy)
model.initialize()
yhat = model.forward(x)
loss = cross_entropy(y, yhat)
print('Before Training - loss value {} accuracy {}'.format(loss, accuracy_score(y, yhat > 0.5)))

#Train network with full-batch gradient descent w fixed learning rate
n_epochs = 150
learning_rate = 0.005
for n in range(n_epochs):
    model.forward(x)
    yhat = model.a[-1]
    model.backward(y, yhat)
    model.update(learning_rate)
    loss = cross_entropy(y, yhat)
    print('Iteration {} - loss value {} accuracy {}'.format(n, loss, accuracy_score(y, yhat > 0.5)))
    
