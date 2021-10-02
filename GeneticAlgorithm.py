from Agent import Agent
from GeneticNeuralNetwork import GeneticNeuralNetwork
from random import random
import math

neural_net_inputs = 64
neural_net_hiddens = 14
neural_net_outputs = 10

genetic_algorithm_mutation_rate = 0.1

#Made specifically for neuroevolutionary purposes
class GeneticAlgorithm(object):
    def __init__(self, _population_size, _training_sets_data, _training_data_sets_results):
        self.generation = 0
        self.population_size = _population_size
        self.agents = []

        self.training_data_sets = _training_sets_data
        self.training_data_sets_results = _training_data_sets_results
    
        for i in range(self.population_size):
            #creating a randomly generated neural net
            neural_net = GeneticNeuralNetwork(neural_net_inputs, neural_net_hiddens, neural_net_outputs)
            #assigning the neural net as the brain of the agent
            agent = Agent(i, neural_net)
            self.agents.append(agent)

    def runEnvironment(self):
        total_score_sum = 0
        for agent in self.agents:
            agent.score = 0
            agent.points = 0
            for i in range(len(self.training_data_sets)):
                training_data = self.training_data_sets[i]

                if i == 0: ##Only run on the first one for performance improvements
                    if len(training_data) != neural_net_inputs:
                        print("Input does not match the form of what was expected.")
                        print("Got:", len(training_data))
                        print("Expected:", neural_net_inputs)

                raw_result = agent.brain.getResult(training_data)
                result = GeneticAlgorithm.evaluateResponse(raw_result)

                actual_result = self.training_data_sets_results[i]
                #print("Guess:", result, "Actual:", actual_result)
                if (result == actual_result):
                    agent.points += 1
                                
            #print("Agent", agent.id, "received", agent.points, "points!")
            agent.score = math.pow(agent.points, 2) #This makes agents who perform well more likely to get chosen,
            total_score_sum += agent.score

        highest_fitness = -1
        highest_fitness_agent = None
        for agent in self.agents:
            current_fitness = agent.score / total_score_sum
            agent.fitness = current_fitness
            if (current_fitness > highest_fitness):
                highest_fitness = current_fitness
                highest_fitness_agent = agent
            
        return highest_fitness_agent

    def performSelection(self):
        newAgents = []
        for i in range(self.population_size):
            #Select an agent from the population
            fitAgent = self.pickFitAgent()
            
            #Copy the brain of the agent, and perform some mutation
            fitBrain = fitAgent.brain.copy()
            fitBrain.mutate(genetic_algorithm_mutation_rate)

            #Create a new agent with the 
            newAgent = Agent(i, fitBrain)
            newAgents.append(newAgent)

        self.agents = newAgents
        
    # Inspired from: https://github.com/CodingTrain/website/blob/main/CodingChallenges/CC_035.4_TSP_GA/P5/ga.js line 54.
    # Pick an agent from the population that has a chance to be picked proportional to its fitness
    def pickFitAgent(self):
        index = 0
        r = random()

        while (r > 0):
            r = r - self.agents[index].fitness
            index += 1
        
        index -= 1
        return self.agents[index]

    def evolveGeneration(self):
        if self.generation != 0:
            self.performSelection()

        fittestAgent = self.runEnvironment()
        self.generation += 1

        return fittestAgent;

        #if self.generation < self.max_generations and self.generation < 950:
        #    self.evolveGeneration()
        # else:
        #     text = input("Do you want to continue? (y/n): ")
        #     shouldIncreaseMaxGenerations = text.lower() == "y"
        #     if (shouldIncreaseMaxGenerations and self.generation < 950): #python will raise a recursion error if you try to reach 1000
        #         self.max_generations += 100
        #         self.evolveGeneration()
        #     else:
        #         print("Complete!")
        #         fittestAgent.brain.writeToJSON()

    ##################### STATIC METHODS  #####################

    @staticmethod
    def evaluateResponse(response):
        #response is made up of 10 results, (numbers 0 through 9)
        max_value = max(response) #getting the nn output with the greatest score
        max_value_index = response.index(max_value)
        return max_value_index #Each number corresponds to it's index in this digit recognition algorithm