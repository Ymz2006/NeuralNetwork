import random

import numpy as np
import matplotlib.pyplot as plt

from dense import Dense
from losses import mse, mse_prime


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 100, learning_rate = 0.01, verbose = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")


test_cases = 1000
input = [random.uniform(0, 1) for _ in range(test_cases)]
output = np.multiply(input,input)
X = np.reshape(input, (test_cases, 1, 1))
Y = np.reshape(output, (test_cases, 1, 1))

network = [
    Dense(1, 8,"tanh"),
    Dense(8, 8, "tanh"),
    Dense(8, 1, "none"),
]

# train
train(network, mse, mse_prime, X, Y, epochs=50, learning_rate=0.1)




predicted = []
for i in input:
     predicted.append(predict(network,[[i]]))


plt.scatter(input,output, color='blue')
plt.scatter(input,predicted, color='red')

plt.show()