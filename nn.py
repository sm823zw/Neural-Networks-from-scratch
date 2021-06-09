import numpy as np

# Layer class
class Layer:

    # Initialization - weights - stores weights, bias - stores bias, activation_fun - stores the activation for that layer
    # error - stores error,delta - stores the 'error' term calculated during backprop
    # momentum_term_w and momentum_term_b - stores the firt order moments, 
    # input_to_activation - stores the output (z) that goes in as input to the activation function
    def __init__(self, n_input, n_neurons, weights=None, bias=None, activation_fun=None):
        self.weights = weights
        lower, upper = -(1.0 / np.sqrt(n_input)), (1.0 / np.sqrt(n_input))
        self.weights = lower + np.random.randn(n_input, n_neurons) * (upper - lower)
        self.bias = 0.0
        self.activation_fun = activation_fun
        self.error = None
        self.delta = None
        self.momentum_term_w = 0.0
        self.momentum_term_b = 0.0
        self.input_to_activation = None

    # One step of Forward propagation
    def activate(self, x):
        h = np.dot(x, self.weights) + self.bias
        self.input_to_activation = self.apply_activation(h)
        return self.input_to_activation
        
    # Apply the given activation function
    def apply_activation(self, x):
        if self.activation_fun == 'relu':
            return np.maximum(x, 0)
        elif self.activation_fun == 'sigmoid':
            return np.exp(-np.logaddexp(0, -x))
        elif self.activation_fun == 'tanh':
            return np.tanh(x)
        elif self.activation_fun == 'softmax':
            exp = np.exp(x - x.max())
            return exp / np.sum(exp, axis=0)
    
    # Apply derivative of the given activation function
    def apply_activation_d(self, x):
        if self.activation_fun == 'relu':
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        elif self.activation_fun == 'sigmoid':
            return x * (1 - x)
        elif self.activation_fun == 'tanh':
            return 1 - x ** 2
        elif self.activation_fun == 'softmax':
            return 1

# Neural Network class
class NeuralNet:

    # Initialization - create a list of Layers
    def __init__(self):
        self.layers = []

    # Function to add layer to the NN
    def add_layer(self, layer):
        self.layers.append(layer)
    
    # Forward propagation through the network
    def forward_prop(self, X):
        for layer in self.layers:
            X = layer.activate(X)
        return X
    
    # Predict
    def predict(self, X):
        return self.forward_prop(X)
    
    # Run Backward propagation, calculate the 'error' terms, and update the weights
    def backward_prop(self, X, y, learning_rate, momentum):
        # Peform forward propagation
        output = self.forward_prop(X)
        # Calculate gradients
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if layer == self.layers[-1]:
                layer.error = output - y
                layer.delta = layer.error*layer.apply_activation_d(output)
            else:
                next_layer = self.layers[i+1]
                layer.error = np.dot(next_layer.weights, next_layer.delta) + np.sum(next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_d(layer.input_to_activation)
        # Update the weights and biases
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                inp = np.atleast_2d(X)
            else:
                inp = np.atleast_2d(self.layers[i-1].input_to_activation)
            layer.momentum_term_w = layer.delta * inp.T * learning_rate + momentum * layer.momentum_term_w
            layer.weights -= layer.momentum_term_w
            layer.momentum_term_b = layer.delta * 1 * learning_rate + momentum * layer.momentum_term_b
            layer.bias -= layer.momentum_term_b

    # Train the model
    def train(self, X, y, learning_rate, momentum, epochs):
        errors = []
        for i in range(epochs):
            for j in range(len(X)):
                self.backward_prop(X[j], y[j], learning_rate, momentum)
            cel = -(1/X.shape[0]) * np.sum(y*np.log(self.forward_prop(X)))
            errors.append(cel)
            print('Epoch: #%s, Loss: %f' % (i, float(cel)))
        return errors

    # Calculate accuracy
    def calc_accuracy(self, X, y):
        y_hat = self.forward_prop(X)
        acc = list(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1)).count(True)/X.shape[0]
        return acc*100
