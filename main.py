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

# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from input
        self.output = np.maximum(0, inputs)
        
# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
    
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Normalize them for each sample
        probabilities  = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities

# Create dataset            
X, y = spiral_data(samples=100, classes=3) # inputs = 300, features = 2 (300,2)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3) # neurons = 3, weights = 2 (3,2)

# Create ReLU activation (to be used with Dense Layer)
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer) and 3 output values
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer)
activation2 = Activation_Softmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation func.
# Takes in output from previous layer
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activation function
# it makes the output of second dense layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples:
print(activation2.output[:5]) # (300,2) x (2,3) = (300,3)