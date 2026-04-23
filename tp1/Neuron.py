import numpy as np


class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(np.dot(inputs, self.weights) + self.bias)
        return self.output

    def backward(self, error, learning_rate):
        d_output = error * self.sigmoid_derivative(self.output)

        self.weights += learning_rate * d_output * self.inputs
        self.bias += learning_rate * d_output

        return d_output * self.weights