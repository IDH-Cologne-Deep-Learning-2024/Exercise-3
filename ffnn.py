import numpy as np
#installed via terminal due to various issues, no problems on my end

# Activation functions
def ReLU(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))  
    return exp_x / np.sum(exp_x)

# FFNN
def FFNN(input_vector, mode="binary"):
    """
    input_vector: numpy array of shape (4,), representing the input layer
    mode: "binary" for binary classification (single output), "multiclass" for multi-class classification (multiple outputs)
    """
    
    W1 = np.array([[0.5, 1.0, -0.8, 0.1],
                   [-0.1, 0.9, 0.4, 0.01]])
    b1 = np.array([0.1, -0.05])

    
    W2 = np.array([[0.8, 0.2],
                   [-0.7, 0.3]])
    b2 = np.array([0.05, -0.3])

    
    if mode == "binary":
        W3 = np.array([[0.9, -0.4]])
        b3 = np.array([-0.01])
    elif mode == "multiclass":
        W3 = np.array([[0.9, -0.3],
                       [-0.3, 0.1],
                       [0.4, 0.8]])
        b3 = np.array([0.01, 0.05, -0.1])
    else:
        raise ValueError("Invalid mode. Choose 'binary' or 'multiclass'.")

   
    
    h1 = ReLU(np.dot(W1, input_vector) + b1)

    
    h2 = ReLU(np.dot(W2, h1) + b2)

    
    if mode == "binary":
        output = sigmoid(np.dot(W3, h2) + b3)  
    elif mode == "multiclass":
        output = softmax(np.dot(W3, h2) + b3)  

    return output

# Test
input_vector = np.array([1.0, 0.5, -0.5, 0.0])

# Test Net1 (Bin)
output_binary = FFNN(input_vector, mode="binary")
print("Binary Output (Network 1):", output_binary)

# Test Net2 (Mult)
output_multiclass = FFNN(input_vector, mode="multiclass")
print("Multi-class Output (Network 2):", output_multiclass)