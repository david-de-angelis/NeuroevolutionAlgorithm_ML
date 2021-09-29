class Agent(object):
    def __init__(self, _id, _brain):
        self.id = _id
        self.brain = _brain
        self.points = 0 #raw result
        self.score = 0 #achieved from performing well in the environment
        self.fitness = 0.0 #relative chance of being picked compared to other agents