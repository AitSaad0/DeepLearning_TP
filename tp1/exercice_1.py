import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])


outputs = np.array([
    [0],
    [1],
    [1],
    [0]
])

layers = [2, 2, 1]

nn = NeuralNetwork(layers, 0.1, 10000)
nn.train(inputs, outputs)

predicted_output = np.array([nn.predict(x) for x in inputs])
print(predicted_output)

rounded = (predicted_output > 0.5).astype(int)
print("Résultat final :")
print(rounded)