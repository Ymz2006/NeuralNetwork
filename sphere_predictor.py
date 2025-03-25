import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from dense import Dense
from losses import mse, mse_prime

cases = 1000
Xraw = []
Yraw = []
mpl.use('TkAgg')
def predict(network, input,p):
    output = input
    for layer in network:
        output = layer.forward(output)

    return output

def train(network, loss, loss_prime, x_train, y_train, epochs, learning_rate = 0.01, verbose = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x,1)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")

def generate_point_in_unit_circle():
    rad = 1
    # Generate random radius and angle
    r = np.sqrt(random.uniform(0, rad))  # sqrt ensures uniform distribution within the circle
    theta = random.uniform(0, 2 * np.pi)  # random angle
    # Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return [x, y] , np.sqrt(rad*rad - x*x - y*y)

for i in range (cases):
    a,b = generate_point_in_unit_circle()
    Xraw.append(a)
    Yraw.append([b])

X = np.reshape(Xraw, (cases, 2, 1))
Y = np.reshape(Yraw, (cases, 1, 1))

network = [
    Dense(2, 20,"sigmoid"),
    Dense(20, 10, "sigmoid"),
    Dense(10, 10, "sigmoid"),
    Dense(10, 1, "none"),

]

# train
train(network, mse, mse_prime, X, Y, epochs=100, learning_rate=0.01)

# decision boundary plot
points = []
for x in np.linspace(0, 1, 30):
    for y in np.linspace(0, 1, 30):
        if x*x + y*y < 1:
            z = predict(network, [[x], [y]],1)
            points.append([x, y, z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()
