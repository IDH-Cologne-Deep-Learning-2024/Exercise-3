import numpy as np

v = np.array([1,2,3])
np.dot()
np.sum()
np.exp()

# Implementation of simple feed forward neural network
def ReLU(x):
    return np.maximum(0,x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x-np.max(x))
    return exp_x / np.sum(exp_x)

def FFNN(input_vector, mode="binary"):
    pass

def FFNN(input_vector, mode="multiclass"):
    pass


def FFNN(input_vector, mode = "binary"):
    if mode == "binary":
        l1 = ReLu(np.dot(weights[0], input_vector) + biases [0])
        l2 = ReLu(np.dot(weights[1], l1) + biases[1])
        output = sigmoid(np.dot(weights.dot[2], l2) + biases[2])
    elif mode == "multiclass":    
        l1 = ReLu(np.dot(weights[0], input_vector) + biases [0])
        l2 = ReLu(np.dot(weights[1], l1) + biases[1])
        output = softmax(np.dot(weights.dot[2], l2) + biases[2])
    else:
        raise ValueError("Invalid mode. Choose 'binary' or 'multiclass'.")
    return output



# completed Python script with explanations:
# python
# import numpy as np

# v = np.array([1,2,3])
# np.dot()  # Used for matrix multiplication
# np.sum()  # Used for summing array elements
# np.exp()  # Used for exponential function

# # Implementation of simple feed forward neural network

# def ReLU(x):
#     return np.maximum(0, x)

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def softmax(x):
#     exp_x = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
#     return exp_x / np.sum(exp_x)

# def FFNN(input_vector, weights, biases, mode="binary"):
#     if mode == "binary":
#         # Binary classification
#         layer1 = ReLU(np.dot(weights[0], input_vector) + biases[0])
#         layer2 = ReLU(np.dot(weights[1], layer1) + biases[1])
#         output = sigmoid(np.dot(weights[2], layer2) + biases[2])
#     elif mode == "multiclass":
#         # Multiclass classification
#         layer1 = ReLU(np.dot(weights[0], input_vector) + biases[0])
#         layer2 = ReLU(np.dot(weights[1], layer1) + biases[1])
#         output = softmax(np.dot(weights[2], layer2) + biases[2])
#     else:
#         raise ValueError("Invalid mode. Choose 'binary' or 'multiclass'.")
    
#     return output
