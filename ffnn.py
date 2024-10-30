#!/usr/bin/env python3

"""
Implementation of simple feed forward neural network
"""
import numpy as np


def ReLU(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0
    return x


def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x


def softmax(x):
    y = np.exp(x - np.max(x))
    return y / y.sum(axis=0)


def FFNN(input_vector, mode="binary"):
    weight1 = np.array([[0.5, 0.1, -0.8, 0.1], [0.9, -0.1, 0.4, -0.4]])
    weight2 = np.array([[0.8, 0.7], [0.2, 0.3]])
    weight3a = np.array([0.9, -0.4])
    weight3b = np.array([[0.9, -0.3], [-0.4, 0.2], [0.3, 0.8]])
    b1 = np.array([0.05, 0.01])
    b2 = np.array([0.05, 0.03])
    b3a = -0.01
    b3b = np.array([-0.01, 0.1, 0.05])

    # layer 1 & 2
    layer1 = np.add(np.array([np.dot(input_vector, weight1[0]), np.dot(input_vector, weight1[1])]), b1)
    activation1 = ReLU(layer1)
    layer2 = np.add(np.array([np.dot(activation1, weight2[0]), np.dot(activation1, weight2[1])]), b2)
    activation2 = ReLU(layer2)
    if mode == "binary":
        return sigmoid(np.dot(activation2, weight3a) + b3a)
    elif mode == "multiclass":
        layer3 = np.add(np.array([np.dot(activation2, weight3b[0]), np.dot(activation2, weight3b[1]), np.dot(activation2, weight3b[2])]), b3b)
        return softmax(layer3)


testarray = np.array([1.0, 2.0, 3.0, 4.0])
outputbinary = FFNN(testarray, mode="binary")
outputmulticlass = FFNN(testarray, mode="multiclass")
print("Test Binary with Input [1, 2, 3, 4]: ", outputbinary)
print("Test Multiclass with Input [1, 2, 3, 4]: ", outputmulticlass)
