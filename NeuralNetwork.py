import math
import json
import numpy as np 
from Matrix import Matrix

#Made to represent a standard Neural Network
class NeuralNetwork(object):
    def __init__(self, _num_inputs, _num_hidden_neurons, _num_outputs, source_neural_net = None):

        if (isinstance(source_neural_net, NeuralNetwork)):
            self.num_inputs = source_neural_net.num_inputs
            self.num_hidden_neurons = source_neural_net.num_hidden_neurons
            self.num_outputs = source_neural_net.num_outputs

            self.synapse_weights_input_hidden = source_neural_net.synapse_weights_input_hidden.copy()
            self.synapse_weights_hidden_output = source_neural_net.synapse_weights_hidden_output.copy()
            
            self.bias_hidden = source_neural_net.bias_hidden.copy()
            self.bias_output = source_neural_net.bias_output.copy()
        else:
            self.num_inputs = _num_inputs
            self.num_hidden_neurons = _num_hidden_neurons
            self.num_outputs = _num_outputs

            #Establishing (random) weights for each of the synapses (connections between neurons)
            self.synapse_weights_input_hidden = Matrix(self.num_hidden_neurons, self.num_inputs)
            self.synapse_weights_input_hidden.randomise()
            
            self.synapse_weights_hidden_output = Matrix(self.num_outputs, self.num_hidden_neurons)
            self.synapse_weights_hidden_output.randomise()

            #Create a bias for each receiving neuron, - this is to counteract inputs of zero (as otherwise we would just receive a zero output)
            self.bias_hidden = Matrix(self.num_hidden_neurons, 1)
            self.bias_hidden.randomise()

            self.bias_output = Matrix(self.num_outputs, 1)
            self.bias_output.randomise()

    def getResult(self, input_array):
        #Convert the input array to a Neural Network compatible Matrix
        input_matrix = Matrix.createFromArray(input_array)

        #Apply the synapse weights to the result from the result previous neuron
        #Add the bias
        #Scale the number between -1 & 1 (also known as an activation layer)
        #Repeat for next layer

        hidden_layer_result_matrix = Matrix.multiplyMatricies(self.synapse_weights_input_hidden, input_matrix)
        hidden_layer_result_matrix.addMatrix(self.bias_hidden)
        #hidden_layer_result_matrix.sigmoidise()

        output_layer_result_matrix = Matrix.multiplyMatricies(self.synapse_weights_hidden_output, hidden_layer_result_matrix)
        output_layer_result_matrix.addMatrix(self.bias_output)
        #output_layer_result_matrix.sigmoidise()
        

        #ignoring the sigmoidisation due to the fact that numbers were reaching over 700, and causing a math overflow

        return output_layer_result_matrix.toArray()

    def copy(self):
        return NeuralNetwork(None, None, None, self)

    def writeToJSON(self):
        structure = {
            "num_inputs": str(self.num_inputs),
            "num_hidden_neurons": str(self.num_hidden_neurons),
            "num_outputs": str(self.num_outputs),
            "synapse_weights_input_hidden": str(self.synapse_weights_input_hidden),
            "synapse_weights_hidden_output": str(self.synapse_weights_hidden_output),
            "bias_hidden": str(self.bias_hidden),
            "bias_output": str(self.bias_output)
        }
        print(json.dumps(structure))