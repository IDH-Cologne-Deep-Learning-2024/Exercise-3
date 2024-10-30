#!/usr/bin/env python3

"""
Implementation of simple feed forward neural network
"""

import numpy as np

# Activation functions
def ReLU(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # For numerical stability
    return exp_x / np.sum(exp_x)

# Feed-Forward Neural Network function
def FFNN(input_vector, mode="binary"):
    # Weights and biases as per the provided network diagrams
    W1 = np.array([[0.5, 0.1, -0.8, -0.1], [0.9, 0.1, 0.4, 0.01]])
    b1 = np.array([0.05, 0.1])
    
    W2 = np.array([[0.8, 0.2], [-0.05, 0.3]])
    b2 = np.array([0.1, -0.1])
    
    if mode == "binary":
        # Network 1
        W3 = np.array([0.9, -0.4])
        b3 = -0.01
    elif mode == "multiclass":
        # Network 2
        W3 = np.array([[0.9, -0.3], [0.1, -0.4], [0.8, 0.05]])
        b3 = np.array([-0.01, 0.05, 0.1])
    
    # Layer 1
    h1 = ReLU(np.dot(W1, input_vector) + b1)
    
    # Layer 2
    h2 = ReLU(np.dot(W2, h1) + b2)
    
    # Output layer
    if mode == "binary":
        # Binary classification (Network 1)
        y = sigmoid(np.dot(W3, h2) + b3)
        return y
    elif mode == "multiclass":
        # Multiclass classification (Network 2)
        y = softmax(np.dot(W3, h2) + b3)
        return y

