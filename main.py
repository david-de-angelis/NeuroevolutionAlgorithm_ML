from GeneticNeuralNetwork import GeneticNeuralNetwork
from Matrix import Matrix
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

training_data_size = 1500

def setup():
    #### Arrange ####
    digits = load_digits()
    n_samples = len(digits.images)
    undefined_data_samples = digits.images.reshape((n_samples, -1))
    results = digits.target
    
    if training_data_size > n_samples - 1:
        print("The training data size that you have selected is too large")

    #Retrieve the first 1500 data samples and results for training
    training_data = undefined_data_samples[:training_data_size] 
    training_results = results[:training_data_size]

    #Retrieve the rest for asserting the algorithm's accuracy
    testing_data = undefined_data_samples[training_data_size:]
    testing_results = results[:training_data_size]

    ####   Act   ####



    ####  Assert ####




    # neural_net = GeneticNeuralNetwork(2, 4, 1)
    # copy = neural_net.copy()
    # copy.mutate(0.5)
    # print(copy.bias_hidden)

setup()