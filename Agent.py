class Agent(object):
    def __init__(self, _id, _brain):
        self.id = _id

        # Variable storing the Genetic Neural Network
        self.brain = _brain 

        #raw result of how many correct guesses were made
        self.points = 0 

        #scaled result proportional to the points (e.g. points^2 or some other method, as determined by the GA)
        self.score = 0 

        #relative chance of being picked compared to other agents in the population
        self.fitness = 0.0 