from src.utils import get_param_or_default
from src.controllers.mate import MATE, TD_ERROR_MODE
import numpy

"""
 Dynamic Reward Incentives for Variable Exchange (DRIVE)
"""
class DRIVE(MATE):

    def __init__(self, params):
        super(DRIVE, self).__init__(params)
        self.mate_mode = TD_ERROR_MODE
        self.tracked_deltas = numpy.zeros(self.nr_agents, dtype=numpy.float32)
        self.token_values = numpy.random.rand(self.nr_agents)

    def get_token_value(self, i):
        return self.token_values[i]

    def update_token_value(self, i, neighborhood):
        own_estimate = numpy.mean(self.reward_buffer[:,i])
        neighborhood_size = len(neighborhood)*1.0
        if neighborhood_size > 0:
            # Track all the reward differences according to Algorithm 2
            self.tracked_deltas[i] += numpy.mean(own_estimate - self.trust_request_matrix[neighborhood, i])

    def update_step(self):
        super(DRIVE, self).update_step()
        self.token_values[:] = self.tracked_deltas*1.0/self.count
        self.tracked_deltas[:] = 0
        self.count = 0

