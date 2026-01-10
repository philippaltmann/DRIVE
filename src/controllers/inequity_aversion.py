from src.utils import get_param_or_default
from src.controllers.actor_critic import ActorCritic
import torch
import numpy

STATIC_MODE = "static"
TD_ERROR_MODE = "td_error"
VALUE_DECOMPOSE_MODE = "value_decompose"
MATE_MODES = [STATIC_MODE, TD_ERROR_MODE, VALUE_DECOMPOSE_MODE]

NO_DEFECT = 0
DEFECT_ALL = 1 # Does not send or receive any acknowledgment messages
DEFECT_RESPONSE = 2 # Sends acknowledgment requests but does not respond to incoming requests 
DEFECT_RECEIVE = 3 # Sends acknowledgment requests but does not receive any responses
DEFECT_SEND = 4 # Receives acknowledgment requests but does send any requests itself

DEFECT_MODES = [NO_DEFECT, DEFECT_ALL, DEFECT_RESPONSE, DEFECT_RECEIVE, DEFECT_SEND]

"""
 Mutual Acknowledgment Token Exchange (MATE)
"""
class InequityAversion(ActorCritic):

    def __init__(self, params):
        super(InequityAversion, self).__init__(params)
        self.alpha = get_param_or_default(params, "ia_alpha", 5)
        self.beta = get_param_or_default(params, "ia_beta", 0.05)

    def prepare_transition(self, joint_histories, joint_action, rewards, next_joint_histories, done, info):
        transition = super(InequityAversion, self).prepare_transition(joint_histories, joint_action, rewards, next_joint_histories, done, info)
        original_rewards = [r for r in rewards]
        for i, reward_i in enumerate(original_rewards):
            alpha_weighted = 0.0
            beta_weighted = 0.0
            normalization = self.nr_agents - 1
            for j, reward_j in enumerate(original_rewards):
                if i != j:
                    alpha_weighted += max(0, reward_j - reward_i)
                    beta_weighted += max(0, reward_i - reward_j)
            transition["rewards"][i] -= alpha_weighted/normalization
            transition["rewards"][i] -= beta_weighted/normalization
        return transition
