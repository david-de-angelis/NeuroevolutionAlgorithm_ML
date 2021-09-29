from NeuralNetwork import NeuralNetwork
from random import random
import numpy as np 

#Made for neuroevolutionary purposes
class GeneticNeuralNetwork(NeuralNetwork):
    def __init__(self, _num_inputs, _num_hidden_neurons, _num_outputs, source_neural_net = None):
        super().__init__(_num_inputs, _num_hidden_neurons, _num_outputs, source_neural_net)

    def copy(self):
        return GeneticNeuralNetwork(None, None, None, self)

    def getMutatedValue(self, value):
        offset = np.random.normal(0, 1) * 0.5
        new_value = value + offset
        #print("Mutated value from", value, "to", new_value)
        return new_value

    def mutate(self, mutationRate):
        if mutationRate > 1 or mutationRate < 0:
            print("Mutation Rate should be betwen 0 and 1")

        _mutate_func = lambda x, i, j: (self.getMutatedValue(x)) if (random() < mutationRate) else x
        self.synapse_weights_input_hidden.map(_mutate_func)
        self.synapse_weights_hidden_output.map(_mutate_func)
        self.bias_hidden.map(_mutate_func)
        self.bias_output.map(_mutate_func)

    