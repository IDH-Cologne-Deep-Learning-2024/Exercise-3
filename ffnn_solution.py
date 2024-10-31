#!/usr/bin/env python3

"""
Implementation of simple feed forward neural network
"""

import numpy as np


def ReLU(x):
    return max(0, x)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def softmax(x):
    exponentials = np.exp(x)
    sum_of_exponentials = np.sum(np.exp(x))
    return exponentials/sum_of_exponentials


def FFNN(input_vector, mode="binary"):
    # Hidden Layer 1
    # Neuron 1
    W_1_1 = np.array([0.5, 0.1, -0.8, 0.1])
    b_1_1 = 0.05
    h_1_1 = ReLU(np.dot(input_vector, W_1_1) + b_1_1)
    # Neuron 2
    W_1_2 = np.array([0.9, -0.1, 0.4, -0.4])
    b_1_2 = 0.01
    h_1_2 = ReLU(np.dot(input_vector, W_1_2) + b_1_2)
    # Vector for hidden layer 1
    h_1 = np.array([h_1_1, h_1_2])

    # Hidden layer 2
    # Neuron 1
    W_2_1 = np.array([0.8, 0.2])
    b_2_1 = -0.05
    h_2_1 = ReLU(np.dot(h_1, W_2_1) + b_2_1)
    # Neuron 2
    W_2_2 = np.array([-0.7, 0.3])
    b_2_2 = 0.3
    h_2_2 = ReLU(np.dot(h_1, W_2_2) + b_2_2)
    # Vector for hidden layer 2
    h_2 = np.array([h_2_1, h_2_2])

    # Output layer
    if mode == "binary":
        W_3 = np.array([0.9, -0.4])
        b_3 = -0.01
        y = sigmoid(np.dot(h_2, W_3) + b_3)
    elif mode == "multiclass":
        W_3_1 = np.array([0.9, -0.4])
        b_3_1 = -0.01
        y_1 = np.dot(h_2, W_3_1) + b_3_1
        W_3_2 = np.array([-0.3, 0.2])
        b_3_2 = 0.1
        y_2 = np.dot(h_2, W_3_2) + b_3_2
        W_3_3 = np.array([-0.3, 0.8])
        b_3_3 = 0.05
        y_3 = np.dot(h_2, W_3_3) + b_3_3
        y = softmax(np.array([y_1, y_2, y_3]))
    else:
        raise ValueError(f"Unexpected argument value for option 'mode' in FFNN(): '{mode}'. Possible values are 'binary' and 'multiclass'")
    return y


def main():
    x_1 = [0, 1, 4.5, 13.2]
    x_2 = [1, 1, 2, 87.2]
    print(FFNN(x_1, mode="binary"))
    print(FFNN(x_2, mode="binary"))

    print(FFNN(x_1, mode="multiclass"))
    print(FFNN(x_2, mode="multiclass"))


if __name__ == "__main__":
    main()
