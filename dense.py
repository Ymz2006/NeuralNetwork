import numpy as np
def tanh(x):
    return np.tanh(x)
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    def relu_val(x1):
        if x1 < 0:
            return 0
        return x1
    return (np.vectorize(relu_val))(x)

def relu_derivative(x):
    def relu_derivative_val(x1):
        if x1 < 0:
            return 0
        return 1
    return (np.vectorize(relu_derivative_val))(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def none(x):
    return x
def none_derivative(x):
    return 1

activation_dict = {
    "tanh": tanh,
    "relu" : relu,
    "sigmoid" : sigmoid,
    "none" : none
}
activation_der_dict = {
    "tanh": tanh_derivative,
    "relu" : relu_derivative,
    "sigmoid" : sigmoid_derivative,
    "none" : none_derivative
}
class Dense:
    def __init__(self, input_size, output_size, act_fun):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.activation = activation_dict[act_fun]
        self.activation_prime = activation_der_dict[act_fun]
    def forward(self, input):
        self.input = input

        self.pre_activation = np.dot(self.weights, self.input) + self.bias
        self.forwarded = self.activation(np.dot(self.weights, self.input) + self.bias)
        return self.forwarded

    def backward(self, output_gradient, learning_rate):


        output_gradient = np.multiply(output_gradient, self.activation_prime(self.pre_activation))

        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
