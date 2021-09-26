from GeneticNeuralNetwork import GeneticNeuralNetwork
from Matrix import Matrix

def setup():
    neural_net = GeneticNeuralNetwork(2, 4, 1)
    copy = neural_net.copy()
    copy.mutate(0.5)
    print(copy.bias_hidden)

setup()