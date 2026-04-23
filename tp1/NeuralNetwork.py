import numpy as np
from Layer import Layer


class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.1, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = []

        for i in range(len(layers) - 1):
            self.layers.append(Layer(layers[i+1], layers[i]))

    def train(self, inputs, outputs):
        for epoch in range(self.epochs):
            total_error = 0

            for x, y in zip(inputs, outputs):
                activations = [x]

                # Forward
                for layer in self.layers:
                    activations.append(layer.forward(activations[-1]))

                # erreur
                output_errors = y - activations[-1]
                total_error += np.sum(output_errors ** 2)

                # Backward
                errors = output_errors
                for i in reversed(range(len(self.layers))):
                    errors = self.layers[i].backward(errors, self.learning_rate)

            if epoch % 1000 == 0:
                mse = total_error / len(inputs)
                print(f"Epoch {epoch}, MSE: {mse}")

    def predict(self, inputs):
        activations = inputs
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations