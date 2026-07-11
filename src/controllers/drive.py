from src.utils import get_param_or_default
from src.controllers.mate import (MATE, NO_DEFECT, DEFECT_ALL, DEFECT_RESPONSE, DEFECT_RECEIVE, DEFECT_SEND)
import numpy
INSTANT_REWARD = "instant"
MEAN_REWARD = "mean"
RESPONSE_MODES = [INSTANT_REWARD, MEAN_REWARD]

"""
 Dynamic Reward Incentives for Variable Exchange (DRIVE)
"""
class DRIVE(MATE):

    def __init__(self, params):
        super(DRIVE, self).__init__(params)
        self.response_mode = params.get("response_mode", MEAN_REWARD)
        self.tracked_deltas = numpy.zeros(self.nr_agents, dtype=numpy.float32)
        self.token_values = numpy.random.rand(self.nr_agents)
        self.reward_capacity = params["episodes_per_epoch"]*params["time_limit"]
        self.reward_buffer = numpy.zeros((self.reward_capacity, self.nr_agents), dtype=numpy.float32)
        self.count = 0

    def get_token_value(self, i):
        return self.token_values[i]

    def update_token_value(self, i, neighborhood):
        if self.response_mode == MEAN_REWARD: own_estimate = numpy.mean(self.reward_buffer[:,i])
        else: own_estimate = self.reward_buffer[-1,i]
        neighborhood_size = len(neighborhood)*1.0
        if neighborhood_size > 0:
            # Track all the reward differences according to Algorithm 2
            self.tracked_deltas[i] += numpy.mean(own_estimate - self.trust_request_matrix[neighborhood, i])

    def update_step(self):
        super(DRIVE, self).update_step()
        self.token_values[:] = self.tracked_deltas*1.0/self.count
        self.tracked_deltas[:] = 0
        self.count = 0

    def save_rewards(self, rewards):
        index = self.count%self.reward_capacity
        self.reward_buffer[index] = numpy.abs(rewards)
        self.count += 1

    def prepare_transition(self, joint_histories, joint_action, rewards, next_joint_histories, done, info):
        self.save_rewards(rewards)
        transition = super(MATE, self).prepare_transition(joint_histories, joint_action, rewards, next_joint_histories, done, info)
        original_rewards = [r for r in rewards]
        self.trust_request_matrix[:] = 0
        self.trust_response_matrix[:] = 0
        defector_id = -1
        if self.defect_mode != NO_DEFECT:
            defector_id = numpy.random.randint(0, self.nr_agents)
        # Whether responder i receives incoming requests at this timestep.
        request_receive_enabled = [self.sample_no_comm_failure() for _ in range(self.nr_agents)]

        # 1.if TD_i(u_t,i) >= 0, send requests to neighbours.
        for i, reward, history, next_history in zip(range(self.nr_agents), original_rewards, joint_histories, next_joint_histories):
            self.update_token_value(i, info["neighbor_agents"][i])
            requests_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_SEND]
            requests_enabled = requests_enabled and self.sample_no_comm_failure()
            if requests_enabled and self.can_rely_on(i, reward, history, next_history, recompute=True):
                neighborhood = info["neighbor_agents"][i]
                self.trust_request_matrix[neighborhood,i] += self.get_token_value(i)
                transition["request_messages_sent"] += len(neighborhood)
        
        # 2. Send trust responses
        for i, history, next_history in zip(range(self.nr_agents), joint_histories, next_joint_histories):
            neighborhood = info["neighbor_agents"][i]
            respond_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_RESPONSE]
            respond_enabled = respond_enabled and self.sample_no_comm_failure()
            # 2.1 Augment own reward if received request 
            if request_receive_enabled[i]:
                trust_requests = self.trust_request_matrix[i, neighborhood]
                if len(trust_requests) > 0:
                    transition["rewards"][i] += numpy.max(trust_requests)
            # 2.2 Compute repsonse 
            if respond_enabled and len(neighborhood) > 0:
                if self.can_rely_on(i, transition["rewards"][i], history, next_history, recompute=False):
                    accept_trust = self.get_token_value(i)
                else: accept_trust = -self.token_value
                for j in neighborhood:
                    assert i != j
                    if self.trust_request_matrix[i][j] > 0:
                        self.trust_response_matrix[j][i] = accept_trust
                        transition["response_messages_sent"] += accept_trust > 0
        
        # 3. Receive trust responses
        for i, trust_responses in enumerate(self.trust_response_matrix):
            neighborhood = info["neighbor_agents"][i]
            receive_enabled = (i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_RECEIVE]) and self.sample_no_comm_failure()
            if receive_enabled and len(neighborhood) > 0 and trust_responses.any():
                filtered_trust_responses = [trust_responses[x] for x in neighborhood if abs(trust_responses[x]) > 0]
                if len(filtered_trust_responses) > 0: transition["rewards"][i] += numpy.min(filtered_trust_responses)

        if done: self.last_rewards_observed = [[] for _ in range(self.nr_agents)]
        return transition