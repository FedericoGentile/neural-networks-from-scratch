import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_features, n_neurons):
        self.weights = 0.01*np.random.rand(n_features, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
X, y = spiral_data(samples=100, classes=3) # inputs = 300, features = 2 (300,2)

dense1 = Layer_Dense(2, 3) # neurons = 3, weights = 2 (3,2)

dense1.forward(X)

print(dense1.output[:5]) # (300,2) x (2,3) = (300,3)