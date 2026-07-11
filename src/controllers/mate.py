from src.utils import get_param_or_default
from src.controllers.actor_critic import ActorCritic
import torch
import numpy

STATIC_MODE = "static"
TD_ERROR_MODE = "td_error"
UNGATED_MODE = "ungated"
MATE_MODES = [STATIC_MODE, TD_ERROR_MODE, UNGATED_MODE]

NO_DEFECT = 0
DEFECT_ALL = 1 # Does not send or receive any messages
DEFECT_RESPONSE = 2 # Sends requests but does not respond to incoming requests
DEFECT_RECEIVE = 3 # Sends requests but does not receive any responses
DEFECT_SEND = 4 # Receives requests but does not send any requests itself

DEFECT_MODES = [NO_DEFECT, DEFECT_ALL, DEFECT_RESPONSE, DEFECT_RECEIVE, DEFECT_SEND]

"""
 Mutual Acknowledgment Token Exchange (MATE)
"""
class MATE(ActorCritic):

    def __init__(self, params):
        super(MATE, self).__init__(params)
        self.last_rewards_observed = [[] for _ in range(self.nr_agents)]
        self.mate_mode = get_param_or_default(params, "mate_mode", STATIC_MODE)
        self.token_value = get_param_or_default(params, "token_value", 1)
        self.trust_request_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=numpy.float32)
        self.trust_response_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=numpy.float32)
        self.defect_mode = get_param_or_default(params, "defect_mode", NO_DEFECT)
        self.current_values = numpy.zeros(self.nr_agents, dtype=numpy.float32)
        self.next_values = numpy.zeros(self.nr_agents, dtype=numpy.float32)
        
    def can_rely_on(self, agent_id, reward, history, next_history, recompute=False):
        if self.mate_mode == UNGATED_MODE: return True
        if self.mate_mode == STATIC_MODE:
            if not self.last_rewards_observed[agent_id]:
                self.last_rewards_observed[agent_id].append(reward)
                return True
            last_reward = numpy.mean(self.last_rewards_observed[agent_id])
            self.last_rewards_observed[agent_id].append(reward)
            return reward >= last_reward
        if self.mate_mode == TD_ERROR_MODE:
            if recompute:
                history = torch.tensor(numpy.asarray([history]), dtype=torch.float32, device=self.device)
                next_history = torch.tensor(numpy.asarray([next_history]), dtype=torch.float32, device=self.device)
                self.current_values[agent_id] = self.get_values(agent_id, history)[0].item()
                self.next_values[agent_id] = self.get_values(agent_id, next_history)[0].item()
            return reward + self.gamma * self.next_values[agent_id] - self.current_values[agent_id] >= 0

    def prepare_transition(self, joint_histories, joint_action, rewards, next_joint_histories, done, info):
        transition = super(MATE, self).prepare_transition(joint_histories, joint_action, rewards, next_joint_histories, done, info)
        original_rewards = [r for r in rewards]
        self.trust_request_matrix[:] = 0
        self.trust_response_matrix[:] = 0
        defector_id = -1
        if self.defect_mode != NO_DEFECT: defector_id = numpy.random.randint(0, self.nr_agents)
        request_receive_enabled = [self.sample_no_comm_failure() for _ in range(self.nr_agents)]

        # 1. Send MATE requests to neighbours if TD_i(u_t,i) >= 0.
        for i, reward, history, next_history in zip(range(self.nr_agents), original_rewards, joint_histories, next_joint_histories):
            requests_enabled = (i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_SEND]) and self.sample_no_comm_failure()
            if requests_enabled and self.can_rely_on(i, reward, history, next_history, recompute=True):
                neighborhood = info["neighbor_agents"][i]
                self.trust_request_matrix[neighborhood, i] += self.token_value
                transition["request_messages_sent"] += len(neighborhood)
        
        # 2. Send MATE responses
        for i, history, next_history in zip(range(self.nr_agents), joint_histories, next_joint_histories):
            neighborhood = info["neighbor_agents"][i]
            respond_enabled = (i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_RESPONSE]) and self.sample_no_comm_failure()
            # 2.1 Augment own reward if received request 
            if request_receive_enabled[i]:
                trust_requests = self.trust_request_matrix[i, neighborhood]
                if len(trust_requests) > 0:
                    transition["rewards"][i] += numpy.max(trust_requests)
            # 2.2 Compute repsonse 
            if respond_enabled and len(neighborhood) > 0:
                if self.can_rely_on(i, transition["rewards"][i], history, next_history, recompute=False):
                    accept_trust = self.token_value
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
                if len(filtered_trust_responses) > 0: 
                    transition["rewards"][i] += numpy.min(filtered_trust_responses)
                    
        if done: self.last_rewards_observed = [[] for _ in range(self.nr_agents)]
        return transition
