from src.utils import get_param_or_default
from src.controllers.mate import (MATE, NO_DEFECT, DEFECT_ALL, DEFECT_RESPONSE, DEFECT_RECEIVE, DEFECT_SEND)
import numpy
INSTANT_REWARD = "instant"
MEAN_REWARD = "mean"
RESPONSE_MODES = [INSTANT_REWARD, MEAN_REWARD]

""" Dynamic Reward Incentives for Variable Exchange (DRIVE)
 Ablations: 
 - mate_mode = UNGATED => always send requests 
 - response_mode = INSTANT => use own instant reward in response (not eoch mean)
 - reciprocal_trust = True => additionally gate response sing (closer to MATE)"""
class DRIVE(MATE):

    def __init__(self, params):
        super(DRIVE, self).__init__(params)
        self.response_mode = params.get("response_mode", MEAN_REWARD)
        self.req_mask = numpy.zeros((self.nr_agents, self.nr_agents), dtype=bool)
        self.res_mask = numpy.zeros((self.nr_agents, self.nr_agents), dtype=bool)
        self.reciprocal_trust = get_param_or_default(params, "reciprocal_trust", False)
        self.tracked_deltas = numpy.zeros(self.nr_agents, dtype=numpy.float32)
        self.token_values = numpy.random.rand(self.nr_agents)
        self.reward_capacity = params["episodes_per_epoch"]*params["time_limit"]
        self.reward_buffer = numpy.zeros((self.reward_capacity, self.nr_agents), dtype=numpy.float32)
        self.count = 0

    def own_reward_estimate(self, i):
        """Return the responder-side reward value used in Delta."""
        if self.count <= 0: return 0.0
        if self.response_mode == MEAN_REWARD:
            return numpy.mean(self.reward_buffer[:self.count, i])
        if self.response_mode == INSTANT_REWARD:
            latest_index = (self.count - 1) % self.reward_capacity
            return float(self.reward_buffer[latest_index, i])        
        raise NotImplemented(f'{self.response_mode} not defined')
  
    def update_step(self):
        super(DRIVE, self).update_step()
        self.token_values[:] = self.tracked_deltas*1.0/self.count
        self.tracked_deltas[:] = 0
        self.reward_buffer[:] = 0
        self.count = 0

    def save_rewards(self, rewards):
        index = self.count%self.reward_capacity
        if self.reciprocal_trust: rewards = numpy.abs(rewards)
        self.reward_buffer[index] = rewards
        self.count += 1

    def prepare_transition(self, joint_histories, joint_action, rewards, next_joint_histories, done, info):
        self.save_rewards(rewards)
        transition = super(MATE, self).prepare_transition(joint_histories, joint_action, rewards, next_joint_histories, done, info)
        original_rewards = [r for r in rewards]
        self.trust_request_matrix[:] = 0
        self.trust_response_matrix[:] = 0
        self.req_mask[:], self.res_mask[:] = False, False
        defector_id = -1
        if self.defect_mode != NO_DEFECT: defector_id = numpy.random.randint(0, self.nr_agents)
        request_receive_enabled = [self.sample_no_comm_failure() for _ in range(self.nr_agents)]

        # 1. Send DRIVE requests to neighbours if TD_i(u_t,i) >= 0.
        for i, reward, history, next_history in zip(range(self.nr_agents), original_rewards, joint_histories, next_joint_histories):
            neighborhood = info["neighbor_agents"][i]
            if self.reciprocal_trust and len(neighborhood) > 0: # Track all the reward differences according to Algorithm 2
                self.tracked_deltas[i] += numpy.mean(self.own_reward_estimate(i) - self.trust_request_matrix[neighborhood, i])
            requests_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_SEND]
            requests_enabled = requests_enabled and self.sample_no_comm_failure()
            if requests_enabled and self.can_rely_on(i, reward, history, next_history, recompute=True):
                self.trust_request_matrix[neighborhood, i] += self.token_values[i] if self.reciprocal_trust else reward
                self.req_mask[neighborhood, i] = True
                transition["request_messages_sent"] += len(neighborhood)
        
        # 2. Send DRIVE responses: Delta_{j,i} = mean_reward_i - request_reward_j.
        for i, history, next_history in zip(range(self.nr_agents), joint_histories, next_joint_histories):
            neighborhood = info["neighbor_agents"][i]
            respond_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_RESPONSE]
            respond_enabled = respond_enabled and self.sample_no_comm_failure()
            # 2.1 For reciprocal trust: Augment own reward if received request 
            if self.reciprocal_trust and request_receive_enabled[i]:
                trust_requests = self.trust_request_matrix[i, neighborhood]
                if len(trust_requests) > 0: transition["rewards"][i] += numpy.max(trust_requests)
            # 2.2 Compute repsonse 
            if respond_enabled and len(neighborhood) > 0 :
                own_estimate = self.own_reward_estimate(i)
                if self.reciprocal_trust:
                  if self.can_rely_on(i, transition["rewards"][i], history, next_history):
                      accept_trust = self.token_values[i] 
                  else: accept_trust = -self.token_value
                for j in neighborhood:
                    assert i != j
                    if self.req_mask[i][j] and not self.reciprocal_trust:
                        self.trust_response_matrix[j][i] = own_estimate - self.trust_request_matrix[i][j]
                        self.res_mask[j][i] = True
                        transition["response_messages_sent"] += 1
                    elif self.trust_request_matrix[i][j] > 0 and self.reciprocal_trust:
                        self.trust_response_matrix[j][i] = accept_trust
                        transition["response_messages_sent"] += accept_trust > 0

        # 3. Shape rewards: subtract own responses to others, add responses to own requests.
        for i, trust_responses in enumerate(self.trust_response_matrix):
            neighborhood = info["neighbor_agents"][i]
            receive_enabled = (i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_RECEIVE]) and request_receive_enabled[i]

            if len(neighborhood) > 0 and not self.reciprocal_trust:
                # Own responses to others' requests: Δ_{j,i}
                own_responses = self.trust_response_matrix[neighborhood, i][self.res_mask[neighborhood, i]]
                own = numpy.min(own_responses) if own_responses.size else 0
                
                # Responses received to own requests: Δ_{i,j}
                received_responses = self.trust_response_matrix[i, neighborhood][self.res_mask[i, neighborhood]]
                ots = numpy.min(received_responses) if receive_enabled and received_responses.size else 0

                transition["rewards"][i] = transition["rewards"][i] - own + ots

            elif receive_enabled and len(neighborhood) > 0 and trust_responses.any():
                filtered_trust_responses = [trust_responses[x] for x in neighborhood if abs(trust_responses[x]) > 0]
                if len(filtered_trust_responses) > 0: transition["rewards"][i] += numpy.min(filtered_trust_responses)

        if done: self.last_rewards_observed = [[] for _ in range(self.nr_agents)]
        return transition