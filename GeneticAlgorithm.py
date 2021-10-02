from Agent import Agent
from random import random
import math

# neural_net_inputs = 64
# neural_net_hiddens = 14
# neural_net_outputs = 10

genetic_algorithm_mutation_rate = 0.1

#Made specifically for neuroevolutionary purposes
class GeneticAlgorithm(object):
    def __init__(self, _create_brain, _population_size, _training_sets_data, _training_data_sets_results):
        self.create_brain = _create_brain
        self.generation = 0
        self.population_size = _population_size
        self.agents = []

        self.training_data_sets = _training_sets_data
        self.training_data_sets_results = _training_data_sets_results
    
        for i in range(self.population_size):
            brain = self.create_brain()
            agent = Agent(i, brain)
            self.agents.append(agent)

    def runEnvironment(self):
        total_score_sum = 0
        for agent in self.agents:
            agent.score = 0
            agent.points = 0
            for i in range(len(self.training_data_sets)):
                training_data = self.training_data_sets[i]

                raw_result = agent.brain.getResult(training_data)
                result = GeneticAlgorithm.evaluateResponse(raw_result)

                actual_result = self.training_data_sets_results[i]
                #print("Guess:", result, "Actual:", actual_result)

                if (result == actual_result):
                    agent.points += 1
                                
            #print("Agent", agent.id, "received", agent.points, "points!")
            agent.score = math.pow(agent.points, 2) #This makes agents who perform well more likely to get chosen,
            total_score_sum += agent.score

        # loop through all of the agents, and make their fitness relative to the scores of all of the other agents, so that the populations fitness adds to 1.
        # essentially makes the fitness the probability of being selected to proceed to the next generation
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

    ##################### STATIC METHODS  #####################

    @staticmethod
    def evaluateResponse(response):
        #response is made up of 10 results, (numbers 0 through 9)
        max_value = max(response) #getting the nn output with the greatest score
        max_value_index = response.index(max_value)
        return max_value_index #Each number corresponds to it's index in this digit recognition algorithm