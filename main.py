from GeneticNeuralNetwork import GeneticNeuralNetwork
from GeneticAlgorithm import GeneticAlgorithm
from sklearn.datasets import load_digits, load_iris
from sklearn.metrics import accuracy_score
import numpy as np
import sys

class App(object):
    def __init__(self):
        self.data_samples = None
        self.data_results = None
        self.training_data_size = None # How many pieces of data we are comparing the neural nets against to find the best agents
        self.testing_data_size = None # How many pieces of data we are comparing the final agent against to establish it's real-world accuracy        
        self.neural_net_population_size = None # How many neural networks to have in each generation
        self.training_generations = None # How many generations to train the neural nets for
        self.create_brain = None # Function to create a NeuralNetwork of a specified topology

    def shuffle(self):
        randomize = np.arange(len(self.data_samples))
        np.random.shuffle(randomize)
        self.data_samples = self.data_samples[randomize]
        self.data_results = self.data_results[randomize]

    def load(self, dataset_string):
        if dataset_string == "digits":
            print("Loading the digits data from sklearn.datasets (Classification: 10 possible classes - base accuracy: 10%)")
            digits = load_digits()
            n_samples = len(digits.images)
            self.data_samples = digits.images.reshape((n_samples, -1))
            self.data_results = digits.target

            self.training_data_size = 100 
            self.testing_data_size = 100
            self.neural_net_population_size = 100
            self.training_generations = 100 

            self.create_brain = lambda: GeneticNeuralNetwork(64, 14, 10)

        elif dataset_string == "iris":
            print("Loading the iris data from sklearn.datasets (Classification: 3 possible classes - base accuracy: 33.3%)")
            irises = load_iris()
            self.data_samples = irises.data
            self.data_results = irises.target

            self.training_data_size = 100 
            self.testing_data_size = 50
            self.neural_net_population_size = 20
            self.training_generations = 50

            self.create_brain = lambda: GeneticNeuralNetwork(4, 6, 3)

        else:
            print("ERR: dataset" + "'" + dataset_string + "' was not recognised.")
            exit()

        #re-arranges the data_samples and data_results in unison, so there are no patterns in the order of data (e.g. in the iris dataset)
        self.shuffle()

        data_samples_size = len(self.data_samples)
        if data_samples_size < 1:
            print("ERR: No data samples were loaded.")

        if self.training_data_size + self.testing_data_size > data_samples_size:
            print("ERR: The training & testing data size that you have selected is too large.")
            print(dataset_string, "only has", data_samples_size, "data samples")
            exit()

        data_sample_size = len(self.data_samples[0])
        neural_net_input_size = int(self.create_brain().num_inputs)
        if neural_net_input_size != data_sample_size:
            print("ERR: Neural net is not configured properly.")
            print("Expected", data_sample_size, "inputs, but the neural net has", neural_net_input_size + ".")
            exit()

    def train(self):
        #### Arrange ####
        print("\nBeginning the training process...")

        # Retrieve the first 1500 data samples and results for training
        training_data = self.data_samples[:self.training_data_size] 
        training_results = self.data_results[:self.training_data_size]

        ####   Act   ####
        ga = GeneticAlgorithm(self.create_brain, self.neural_net_population_size, training_data, training_results)
        
        global_fittest_agent_points = -1
        global_fittest_agent = None
        while True:
            # evolve the agent, and retrieve the fittest agent from the population
            population_fittest_agent = ga.evolveGeneration()

            # store the fittest agent from the population
            population_fittest_agent_points = population_fittest_agent.points
            if (population_fittest_agent_points > global_fittest_agent_points):
                global_fittest_agent = population_fittest_agent
                global_fittest_agent_points = population_fittest_agent_points
            
            print("Evolved to generation", str(ga.generation) , 
                "| Global max training accuracy:", str(int(global_fittest_agent_points/self.training_data_size * 100)) + "%", 
                "| Population max training accuracy:", str(int(population_fittest_agent_points/self.training_data_size * 100)) + "%")

            if ga.generation >= self.training_generations:
                text = input("Do you want to continue evolving the population (+100 generations)? (y/n): ")
                shouldIncreaseMaxGenerations = text.lower() == "y"
                if (shouldIncreaseMaxGenerations):
                    self.training_generations += 100
                else:
                    break

        
        return global_fittest_agent

    def test(self, agent):
        ####  Assert ####
        print("\nBeginning the testing process...")

        #Retrieve the not-trained-on data for asserting the algorithm's accuracy
        testing_data = self.data_samples[self.training_data_size: self.training_data_size + self.testing_data_size]
        testing_actual_results = self.data_results[self.training_data_size: self.training_data_size + self.testing_data_size].tolist()

        #Retrieving the best trained agent's predictions of the training data
        testing_predicted_results = []
        for i in range(len(testing_data)):
            neural_net_output = agent.brain.getResult(testing_data[i])
            testing_predicted_result = GeneticAlgorithm.evaluateResponse(neural_net_output)
            testing_predicted_results.append(testing_predicted_result)

        # using the sklearn.metrics.accuracy_score function to establish the accuracy of the best trained agent  
        accuracy = accuracy_score(testing_actual_results, testing_predicted_results) * 100
        print("The best trained agent received an actual accuracy of:", str(int(accuracy)) + "%")
        #global_fittest_agent.brain.writeToJSON()

dataset = "iris" #default value
num_arguments = len(sys.argv)
if (num_arguments == 2):
    dataset = str(sys.argv[1])
elif (num_arguments > 2):
    print("ERR: Unexpected parameters provided, try 'python3 main.py iris'")

app = App()
app.load(dataset)
fittest_agent = app.train()
app.test(fittest_agent)