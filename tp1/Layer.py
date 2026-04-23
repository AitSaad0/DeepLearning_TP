import numpy as np
from Neuron import Neuron

class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def forward(self, inputs):
        self.outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])
        return self.outputs

    def backward(self, errors, learning_rate):
        new_errors = np.zeros(len(self.neurons[0].weights))

        for i, neuron in enumerate(self.neurons):
            error = errors[i]
            new_errors += neuron.backward(error, learning_rate)

        return new_errors