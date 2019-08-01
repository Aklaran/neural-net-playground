import numpy as np

class NeuralNetwork:

    def __init__(self, layerSizes):
        # shape of each weight matrix, which correspond to the relative sizes of
        # the adjacent layers which the weights connect
        weightShapes = [(height, width) for height, width in zip(layerSizes[1:],layerSizes[:-1])]

        # Weights determine the slope of each decision boundary.
        # list containing the actual weights of each neural connection
        # initialized as random numbers in a normal distribution, 
        # divided by the square of the size of the layer to normalize for large layer sizes.
        # Normalization is done to ensure the inputs correspond to large slopes in activation function.
        # Large slopes means the network will learn faster (not sure why)
        # These weights will change as the network discovers the correct function.
        self.weights = [np.random.standard_normal(shape)/shape[1]**.5 for shape in weightShapes]

        # Biases allow translation of decision boundaries.
        # Must be added in every layer except input layer, 
        # stored as a column vector of the same size as the layer.
        # initialized to a zero vector
        self.biases = [np.zeros((size,1)) for size in layerSizes[1:]]

    # feed the input through the network
    # output the networks perdictions
    def predict(self, activation):
        # iterate through each layer as represented by the weight connections between layers
        # and associated biases
        for weight, bias in zip(self.weights, self.biases):
            # calculate the activation for this layer by multiplying the current weight matrix
            # by the previous activation and adding the bias.
            # At first iteration, activation is just the inputs.
            activation = self.activation(np.matmul(weight, activation) + bias)
        return activation

    # activation function, which allows the network to make non-linear decision boundaries.
    @staticmethod
    def activation(x):
        # returns a simple sigmoid function
        return 1/(1+np.exp(-x))