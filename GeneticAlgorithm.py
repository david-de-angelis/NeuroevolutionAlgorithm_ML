from Agent import Agent
from GeneticNeuralNetwork import GeneticNeuralNetwork

neural_net_inputs = 64
neural_net_hiddens = 8
neural_net_outputs = 10

#Made specifically for neuroevolutionary purposes
class GeneticAlgorithm(object):
    def __init__(self, _population_size, _max_generations, _training_sets_data, _training_data_sets_results):
        self.generation = 0
        self.population_size = _population_size
        self.max_generations = _max_generations
        self.agents = []

        self.training_data_sets = _training_sets_data
        self.training_data_sets_results = _training_data_sets_results
    
        for i in range(self.population_size):
            #creating a randomly generated neural net
            neural_net = GeneticNeuralNetwork(neural_net_inputs, neural_net_hiddens, neural_net_outputs)
            #assigning the neural net as the brain of the agent
            agent = Agent(i, neural_net)
            self.agents.append(agent)

    def evaluateResponse(self, response):
        #response is made up of 10 results, (numbers 0 through 9)
        max_value = max(response)
        max_value_index = response.index(max_value)
        return max_value_index #Each number corresponds to it's index in this digit recognition algorithm

    def evolveGeneration(self):
        for agent in self.agents:
            agent.score = 0
            for i in range(len(self.training_data_sets)):
                training_data = self.training_data_sets[i]

                if i == 0: ##Only run on the first one for performance improvements
                    if len(training_data) != neural_net_inputs:
                        print("Input does not match the form of what was expected.")
                        print("Got:", len(training_data))
                        print("Expected:", neural_net_inputs)

                raw_result = agent.brain.getResult(training_data)
                result = self.evaluateResponse(raw_result)

                actual_result = self.training_data_sets_results[i]
                if (result == actual_result):
                    agent.score += 1

            print("Agent", agent.id, "received score", agent.score)


                
                

        self.generation += 1
        print("Evolved to generation " + {self.generation} + "!")

        

    
