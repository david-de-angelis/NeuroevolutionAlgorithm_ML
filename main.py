from GeneticNeuralNetwork import GeneticNeuralNetwork
from GeneticAlgorithm import GeneticAlgorithm
from Matrix import Matrix
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
n_samples = len(digits.images)
data_samples = digits.images.reshape((n_samples, -1))
data_results = digits.target

neural_net_population_size = 100 #100 # How many neural networks to have in each generation
training_generations = 500 #250 # How many generations to train the neural nets for
training_data_size = 200 #1500 # How many pieces of data we are comparing the neural nets against

if training_data_size > n_samples - 1:
    print("The training data size that you have selected is too large")
        
def train():
    #### Arrange ####
    
    #Retrieve the first 1500 data samples and results for training
    training_data = data_samples[:training_data_size] 
    training_results = data_results[:training_data_size]

    #Retrieve the rest for asserting the algorithm's accuracy
    testing_data = data_samples[training_data_size:]
    testing_results = data_results[:training_data_size]

    ga = GeneticAlgorithm(neural_net_population_size,training_generations, training_data, training_results)
    ga.evolveGeneration()

def loop():
    print()
    ####   Act   ####




    ####  Assert ####




    # neural_net = GeneticNeuralNetwork(64, 8, 10)
    # copy = neural_net.copy()
    # copy.mutate(0.5)
    # print(copy.bias_hidden)

train()
