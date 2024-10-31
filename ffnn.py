#!/usr/bin/env python3

"""
Implementation of simple feed forward neural network
"""

import numpy as np

#weights
w1 = np.array([[0.5, 0.1], [-0.8, 0.1],[0.9, -0.1], [0.4, -0.4]])
w2 = np.array([[0.8, 0.2],[-0.7, 0.3]])
w3 = np.array([[0.9],[-0.4]])
#biases
b1 = np.array([[0.05], [0.01]])
b2 = np.array([[-0.05],[0.3]])
b3 = np.array([[-0.01]])

def ReLU(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

#oof unsure
def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def FFNN(input_vector, mode="binary"):
    x = input_vector
    
    hl1 = ReLU(np.dot(x, w1)+b1)
    x = hl1
    hl2 = ReLU(np.dot(x, w2)+b2)
    out = np.dot(hl2, w3)+b3

    if mode == "binary":
        #was supposed to return probability as one value
        out = np.dot(hl2, w3) + b3
        return sigmoid(out)
    elif mode == "multiclass":
        #returns probability as vector
        out1 = np.dot(hl1, w3) + b2
        return softmax(out)

input = np.array([0,2,1,4])
FFNN(input, mode = "binary")
#def need to come back and redo