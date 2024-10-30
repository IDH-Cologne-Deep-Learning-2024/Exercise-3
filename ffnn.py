#!/usr/bin/env python3

"""
Implementation of simple feed forward neural network
"""

import numpy as np

# ReLU activation function
def ReLU(x):
    return np.maximum(0, x)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Softmax activation function
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stability trick
    return exp_x / np.sum(exp_x)


def FFNN(input_vector, mode="binary"):
    # Weights and biases from the diagram
    # First weight matrix W1 (input to first hidden layer)
    W1 = np.array([[0.5, 0.1, -0.8, -0.1],
                   [0.9, -0.1, 0.4, 0.01]])
    b1 = np.array([0.1, 0.2])

    # Second weight matrix W2 (first to second hidden layer)
    W2 = np.array([[0.8, 0.2],
                   [-0.7, 0.3]])
    b2 = np.array([0.3, -0.2])

    # Third weight matrix W3 (second hidden layer to output layer)
    if mode == "binary":
        W3 = np.array([[0.9, -0.4]])
        b3 = np.array([-0.01])
    elif mode == "multiclass":
        W3 = np.array([[0.9, -0.3],
                       [-0.3, 0.4],
                       [0.8, 0.05]])
        b3 = np.array([0.01, -0.02, 0.03])

    # Forward pass
    # 1. Input layer to first hidden layer
    h1 = ReLU(np.dot(W1, input_vector) + b1)

    # 2. First hidden layer to second hidden layer
    h2 = ReLU(np.dot(W2, h1) + b2)

    # 3. Second hidden layer to output layer
    output = np.dot(W3, h2) + b3

    # Output depends on mode
    if mode == "binary":
        return sigmoid(output)  # single probability for Network 1
    elif mode == "multiclass":
        return softmax(output)  # probability vector for Network 2


# Example input
input_vector = np.array([1.0, 0.5, -0.5, 0.2])

# Test Network 1 (Binary Classification)
output_binary = FFNN(input_vector, mode="binary")
print("Binary output:", output_binary)

# Test Network 2 (Multiclass Classification)
output_multiclass = FFNN(input_vector, mode="multiclass")
print("Multiclass output:", output_multiclass)