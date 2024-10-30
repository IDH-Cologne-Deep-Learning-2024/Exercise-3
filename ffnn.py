import numpy as np
def ReLU(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)

def layer(inputV, weights, bias):
    weighted_input = np.dot(inputV, weights)
    return np.add(weighted_input, bias)

def FFNN(input_vector, mode="binary"):

    
    # Weights and biases for Network 1 (Binary classification)
    w1 = np.array([[0.5, 0.1], [0.9, -0.8], [-0.1, 0.4], [0.05, 0.01]])  
    w2 = np.array([[0.8, -0.7], [0.2, 0.3]])  
    w3Bi = np.array([0.9, -0.4]) 
    w3Multi = np.array([[0.9, -0.3, 0.1], [-0.3, 0.4, 0.05]]) 
    b1 = np.array([0.5, 0.1])  
    b2 = np.array([0.3, -0.05])  
    b3Bi = -0.01 
    b3Multi = np.array([0.01, 0.8, -0.01])  

   
    if mode == "binary":
        layer1Bi = ReLU(layer(input_vector, w1, b1))
        layer2Bi = ReLU(layer(layer1Bi, w2, b2))
        outputBi = sigmoid(layer(layer2Bi, w3Bi, b3Bi))
        return outputBi

  
    elif mode == "multiclass":
        layer1Multi = ReLU(layer(input_vector, w1, b1))
        layer2Multi = ReLU(layer(layer1Multi, w2, b2))
        # Output Layer (Multiclass classification, use softmax)
        outputMulti = softmax(layer(layer2Multi, w3Multi, b3Multi))
        return outputMulti

    else:
        raise ValueError("Invalid mode. Choose either 'binary' or 'multiclass'.")


input_vector = np.array([1, 0, 0, 1])


bi_output = FFNN(input_vector)
print("Binary Output withg sigmoid-activation:", bi_output)


multi_output = FFNN(input_vector,"multiclass")
print("Multiclass Output with softmax-activation:", multi_output)
