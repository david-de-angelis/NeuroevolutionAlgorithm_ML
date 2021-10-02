from GeneticNeuralNetwork import GeneticNeuralNetwork
from GeneticAlgorithm import GeneticAlgorithm
from Matrix import Matrix
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

digits = load_digits()
n_samples = len(digits.images)
data_samples = digits.images.reshape((n_samples, -1))
data_results = digits.target
training_data_size = 100 #1500 # How many pieces of data we are comparing the neural nets against
testing_data_size = 100

def train():
    #### Arrange ####
    neural_net_population_size = 50 #100 # How many neural networks to have in each generation
    training_generations = 1000 #250 # How many generations to train the neural nets for

    if training_data_size > n_samples - 1:
        print("The training data size that you have selected is too large")
    
    #Retrieve the first 1500 data samples and results for training
    training_data = data_samples[:training_data_size] 
    training_results = data_results[:training_data_size]

    ####   Act   ####
    ga = GeneticAlgorithm(neural_net_population_size, training_data, training_results)
    
    global_fittest_agent_points = -1
    global_fittest_agent = None
    while True:
        population_fittest_agent = ga.evolveGeneration()

        population_fittest_agent_points = population_fittest_agent.points
        if (population_fittest_agent_points > global_fittest_agent_points):
            global_fittest_agent = population_fittest_agent
            global_fittest_agent_points = population_fittest_agent_points
        
        print("Evolved to generation", str(ga.generation) , "| Global max score:", str(global_fittest_agent_points), "| Population max score:", str(population_fittest_agent_points))

        if ga.generation >= training_generations:
            text = input("Do you want to continue? (y/n): ")
            shouldIncreaseMaxGenerations = text.lower() == "y"
            if (shouldIncreaseMaxGenerations):
                training_generations += 100
            else:
                break

    #global_fittest_agent.brain.writeToJSON()
    return global_fittest_agent

def test(agent):
    #Retrieve the not-trained-on data for asserting the algorithm's accuracy
    testing_data = data_samples[training_data_size: training_data_size+testing_data_size]
    testing_actual_results = data_results[training_data_size:training_data_size+testing_data_size].tolist()

    testing_predicted_results = []
    for i in range(len(testing_data)):
        neural_net_output = agent.brain.getResult(testing_data[i])
        testing_predicted_result = GeneticAlgorithm.evaluateResponse(neural_net_output)
        testing_predicted_results.append(testing_predicted_result)

    accuracy = accuracy_score(testing_actual_results, testing_predicted_results)
    print("The best trained agent recieved an accuracy of:", str(int(accuracy * 100)) + "%")

fittest_agent = train()
test(fittest_agent)