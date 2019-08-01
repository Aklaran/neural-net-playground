import NeuralNetwork as nn
import numpy as np

# number of neurons in each layer
# first layer: input layer
# middle layers: hidden layers
# last layer: output layer
layerSizes = [3000, 500, 10]
net = nn.NeuralNetwork(layerSizes)

# a dummy input representing 10 possible choices
dummyInput = np.ones((layerSizes[0],1))

# use the created neural network to predict the probability of the true answer
# being any one of the elements in input
prediction = net.predict(dummyInput)
print(prediction)