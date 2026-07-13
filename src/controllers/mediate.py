import uuid
from src.utils import get_param_or_default
from src.controllers.actor_critic import ActorCritic
import torch
import numpy
import random

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
Mutually Endorsed Distributed Incentive Acknowledgment Token Exchange (MEDIATE)
Accoding to: https://www.scitepress.org/Papers/2025/130919/130919.pdf
"""
class MEDIATE(ActorCritic):

    def __init__(self, params):
        super(MEDIATE, self).__init__(params)
        self.last_rewards_observed = [[] for _ in range(self.nr_agents)]
        self.mate_mode = get_param_or_default(params, "mate_mode", STATIC_MODE)
        self.defect_mode = get_param_or_default(params, "defect_mode", NO_DEFECT)
        
        #INFO: Default is AutoMATE with Synchronized Consensus
        self.token_value = [get_param_or_default(params, "token_value", 0.1)[i] if isinstance(get_param_or_default(params, "token_value", 0.1), list) else get_param_or_default(params, "token_value", 0.1) for i in range(self.nr_agents)]
        self.no_sync = get_param_or_default(params, "no_sync", False) #True Sovereign Consensus
        self.common_token = get_param_or_default(params, "token_value", [0.1 for _ in range(self.nr_agents)]) #Shared token for Sovereign Consensus
        self.with_consensus = get_param_or_default(params, "consensus_on", True) #False for pure AutoMATE without consensus
        #Baseline Variants:
        self.random_mode = get_param_or_default(params, "random_mode", None) #random token (central from [0.25, 0.5, 1, 2, 4] per "epoch" or "time_step")
        self.fixed_token_mode = get_param_or_default(params, "fixed_token_mode", False) #fixed token
        #Other:
        self.ucb_mode = get_param_or_default(params, "ucb_mode", None) #UCB centralized / decentralized / None
        self.architecture = get_param_or_default(params, "architecture", "holding") #reflecting / centralized 
        
        #MATE Basics
        self.trust_request_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=float)
        self.trust_response_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=float)
        
        #AutoMATE Extensions
        self.values = numpy.zeros(self.nr_agents, dtype=float)
        self.epoch_values = [[] for _ in range(self.nr_agents)]
        self.last_values = [[] for _ in range(self.nr_agents)]
        self.min_reward = [0 for _ in range(self.nr_agents)]
        self.max_reward = [-numpy.inf for _ in range(self.nr_agents)]
        self.rewards = [[] for _ in range(self.nr_agents)]
        self.epoch_returns = numpy.zeros(self.nr_agents, dtype=float)
        self.update_rate = [[] for _ in range(self.nr_agents)]
        self.time_step = 0
        self.episode = 0
        
        #Consensus Extensions
        self.token_send_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=float)
        self.token_response_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=float)
        self.token_shares = [[] for _ in range(self.nr_agents)]
        self.share_list = [[] for _ in range(self.nr_agents)]
        self.first_exchange = [False for _ in range(self.nr_agents)]
        self.share_id = [0 for _ in range(self.nr_agents)]
        
        #UCB 
        self.best_value = [0 for _ in range(self.nr_agents)]
        self.tokens_dict = [{} for _ in range(self.nr_agents)]
        self.last_token_value = [random.choice([0.25, 0.5, 1.0, 2.0, 4.0]) for _ in range(self.nr_agents)]
        self.token_range = get_param_or_default(params, "token_range", [0.25, 0.5, 1.0, 2.0, 4.0])

    def can_rely_on(self, agent_id, reward, history, next_history, value=None, next_value=None):
        if self.mate_mode == STATIC_MODE:
            is_empty = self.last_rewards_observed[agent_id]
            if is_empty:
                self.last_rewards_observed[agent_id].append(reward)
                return True
            last_reward = numpy.mean(self.last_rewards_observed[agent_id])
            self.last_rewards_observed[agent_id].append(reward)
            return reward >= last_reward
        if self.mate_mode == TD_ERROR_MODE:
            if value is None:
                value = self.value_for_history(agent_id, history)
            if next_value is None:
                next_value = self.value_for_history(agent_id, next_history)
            return reward + self.gamma*next_value - value >= 0
        if self.mate_mode == VALUE_DECOMPOSE_MODE:
            return False

    def value_for_history(self, agent_id, history):
        history = torch.as_tensor(numpy.asarray([history]), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            return self.get_values(agent_id, history)[0].item()

    def step_values(self, joint_histories, next_joint_histories):
        values = []
        next_values = []
        with torch.inference_mode():
            for i, (history, next_history) in enumerate(zip(joint_histories, next_joint_histories)):
                if self.mate_mode == TD_ERROR_MODE:
                    history_batch = torch.as_tensor(
                        numpy.asarray([history, next_history]),
                        dtype=torch.float32,
                        device=self.device)
                    value_batch = self.get_values(i, history_batch).view(-1)
                    values.append(value_batch[0].item())
                    next_values.append(value_batch[1].item())
                else:
                    history = torch.as_tensor(numpy.asarray([history]), dtype=torch.float32, device=self.device)
                    values.append(self.get_values(i, history)[0].item())
                    next_values.append(None)
        return values, next_values
    
    #consensus helper method
    def generate_token_shares(self, neighborhood_size, total):
        lower_bound = -total
        upper_bound = +total
        shares = [random.uniform(lower_bound, upper_bound) for _ in range(neighborhood_size)]
        last_share = total - sum(shares)
        shares.append(last_share)
        return shares

    #consensus helper method
    def generate_id(self):
        id = uuid.uuid4()
        return id



    def prepare_transition(self, joint_histories, joint_action, rewards, next_joint_histories, done, info):
        transition = super(MEDIATE, self).prepare_transition(joint_histories, joint_action, rewards, next_joint_histories, done, info)
        original_rewards = [r for r in rewards]
        current_values, next_values = self.step_values(joint_histories, next_joint_histories)
        
        self.trust_request_matrix[:] = 0
        self.trust_response_matrix[:] = 0
        self.epoch_returns += rewards
        
        # AutoMATE
        for i in range(self.nr_agents):
            for r in rewards:
                if r != 0 and not r in self.rewards[i]:
                    self.rewards[i].append(r)
                    
        #Random per time step (default is per epoch)
        if self.random_mode == "time_step":
            if self.architecture == "centralized":
                xtr = random.choice([0.25, 0.5, 1, 2, 4])
                for i in range(self.nr_agents):
                    self.token_value[i] = xtr
            else:
                for i in range(self.nr_agents):
                    self.token_value[i] = random.choice([0.25, 0.5, 1, 2, 4])

        # 1. Send trust requests
        defector_id = -1
        if self.defect_mode != NO_DEFECT:
            defector_id = numpy.random.randint(0, self.nr_agents)
        request_receive_enabled = [self.sample_no_comm_failure() for _ in range(self.nr_agents)]
        
        ### consensus extension
        if self.with_consensus and self.first_exchange[i]:
            self.token_send_matrix[:] = 0
            self.token_response_matrix[:] = 0
            self.token_shares[i] = []
            self.share_list = [[] for _ in range(self.nr_agents)]   
        ### consensus extension
        
        for i, reward, history, next_history in zip(range(self.nr_agents), original_rewards, joint_histories, next_joint_histories):
            self.values[i] += current_values[i]
            neighborhood = info["neighbor_agents"][i]
            
            ### consensus extension
            if done and self.with_consensus and self.first_exchange[i]:
                self.token_shares[i] = self.generate_token_shares(len(neighborhood), self.token_value[i])
                self.token_send_matrix[i][i] = self.token_shares[i][0]
            ### consensus extension
            
            requests_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_SEND]
            requests_enabled = requests_enabled and self.sample_no_comm_failure()
            request_can_rely = None
            if requests_enabled and self.mate_mode == TD_ERROR_MODE:
                request_can_rely = self.can_rely_on(i, reward, history, next_history, current_values[i], next_values[i])
            needs_consensus_shares = done and self.with_consensus and self.first_exchange[i]
            next_index = 1
            for j in neighborhood:
                if requests_enabled:
                    assert i != j
                    
                    ### consensus extension
                    if needs_consensus_shares: 
                        self.token_send_matrix[j][i] = self.token_shares[i][next_index]
                        next_index += 1
                    ### consensus extension
                    
                    if request_can_rely is None:
                        can_rely = self.can_rely_on(i, reward, history, next_history, current_values[i], next_values[i])
                    else:
                        can_rely = request_can_rely
                    if can_rely: # Analyze the "winners" of that step
                        if self.no_sync: #Sovereign Consensus
                            self.trust_request_matrix[j][i] += self.common_token[i]
                        else: 
                            self.trust_request_matrix[j][i] += self.token_value[i]
                        transition["request_messages_sent"] += 1
            
        # 2. Send trust responses
        for i, history, next_history in zip(range(self.nr_agents), joint_histories, next_joint_histories):
            
            ### consensus extension
            if done and self.with_consensus and self.first_exchange[i]:
                summed_token_shares = sum(self.token_send_matrix[i])                                         
                share_id = self.generate_id()
                self.share_list[i].append((summed_token_shares, share_id))
                self.first_exchange[i] = False
            ### consensus extension
            
            neighborhood = info["neighbor_agents"][i]
            respond_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_RESPONSE]
            respond_enabled = respond_enabled and self.sample_no_comm_failure()
            if request_receive_enabled[i]:
                trust_requests = [self.trust_request_matrix[i][x] for x in neighborhood]
                if len(trust_requests) > 0:
                    transition["rewards"][i] += numpy.max(trust_requests)
            if respond_enabled and len(neighborhood) > 0:
                if self.can_rely_on(i, transition["rewards"][i], history, next_history, current_values[i], next_values[i]):
                    if self.no_sync: #Sovereign Consensus
                        accept_trust = self.common_token[i]
                    else: 
                        if self.architecture == "reflecting":
                            accept_trust = self.token_value[j]
                        else:
                            accept_trust = self.token_value[i]
                else:
                    if self.no_sync: #Sovereign Consensus
                        accept_trust = -self.common_token[i]
                    else:
                        if self.architecture == "reflecting":
                            accept_trust = -self.token_value[j]
                        else:
                            accept_trust = -self.token_value[i]
                   
                for j in neighborhood:
                    assert i != j
                    
                    ### consensus extension
                    if self.with_consensus and len(self.share_list[i])>0:
                        for x in self.share_list[i]:
                            if x not in self.share_list[j]:
                                self.share_list[j].append(x)
                    ### consensus extension
                    
                    if self.trust_request_matrix[i][j] > 0:
                        self.trust_response_matrix[j][i] = accept_trust
                        if accept_trust > 0:
                            transition["response_messages_sent"] += 1

        # 3. Receive trust responses
        for i, trust_responses in enumerate(self.trust_response_matrix):
            neighborhood = info["neighbor_agents"][i]
            receive_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_RECEIVE]
            receive_enabled = receive_enabled and self.sample_no_comm_failure()
            if receive_enabled and len(neighborhood) > 0:
                
                ### consensus extension
                if self.with_consensus:
                    if len(self.share_list[i]) > 0:
                        if self.no_sync: #Sovereign Consensus
                            self.common_token[i] = sum([x[0] for x in self.share_list[i]])/len(self.share_list[i])
                        else: 
                            self.token_value[i] = sum([x[0] for x in self.share_list[i]])/len(self.share_list[i])
                ### consensus extension
                
                if trust_responses.any():
                    filtered_trust_responses = [trust_responses[x] for x in neighborhood if abs(trust_responses[x]) > 0]
                    if len(filtered_trust_responses) > 0:
                        transition["rewards"][i] += min(filtered_trust_responses)
        
        self.time_step += 1
        if done:
            
            if self.fixed_token_mode:
                for i in range(self.nr_agents):
                    # transition["values"][i].append(self.values[i]/self.time_step)
                    self.values[i] = 0
                self.time_step = 0
            elif self.random_mode == "epoch":
                random_token = random.choice([0.25, 0.5, 1, 2, 4])
                for i in range(self.nr_agents):
                    self.token_value[i] = random_token
            elif self.random_mode == "time_step":
                pass
            elif self.ucb_mode == 'centralized':
                self.episode += 1
                if self.episode % 10 == 1:
                    max_upper_bound = -numpy.inf
                    if(str(self.last_token_value[0]) not in self.tokens_dict[0]):
                        self.tokens_dict[0][str(self.last_token_value[0])] = {'rewards': [],} 
                    self.tokens_dict[0][str(self.last_token_value[0])]['rewards'].append([self.episode, sum(self.epoch_returns)])
                         
                    for token, stats in self.tokens_dict[0].items():
                        if(len(stats['rewards']) > 0):
                            sum_rewards = sum([x[1] for x in stats['rewards']])
                            mean_reward = sum_rewards / len(stats['rewards'])
                            di = numpy.sqrt((2 * numpy.log(self.episode)) / len(stats['rewards']))
                            upper_bound = mean_reward + di

                        else:
                            upper_bound = 1e400
                        if(upper_bound > max_upper_bound):
                            max_upper_bound = upper_bound
                            self.best_value[0] = float(token)
                    
                    
                    for i in range(self.nr_agents):
                        self.token_value[i] = self.best_value[0]   
                        if self.episode-1 < len(self.token_range)*10:
                            index = int(self.episode / 10)
                            self.token_value[i] = self.token_range[index]
                    self.last_token_value[0] = self.token_value[0]
                    self.epoch_returns = numpy.zeros(self.nr_agents, dtype=float) # reset
                             
            elif self.ucb_mode == 'decentralized':
                self.episode += 1
                if self.episode % 10 == 1:
                    for i in range(self.nr_agents):
                        max_upper_bound = -numpy.inf
                        if(str(self.last_token_value[i]) not in self.tokens_dict[i]):
                            self.tokens_dict[i][str(self.last_token_value[i])] = {'rewards': [],} 
                        self.tokens_dict[i][str(self.last_token_value[i])]['rewards'].append([self.episode, self.epoch_returns[i]])

                        for token, stats in self.tokens_dict[i].items():
                            if(len(stats['rewards']) > 0):
                                sum_rewards = sum([x[1] for x in stats['rewards']])
                                mean_reward = sum_rewards / len(stats['rewards'])
                                di = numpy.sqrt((2 * numpy.log(self.episode)) / len(stats['rewards']))
                                upper_bound = mean_reward + di
                            else:
                                upper_bound = 1e400
                            if(upper_bound > max_upper_bound):
                                max_upper_bound = upper_bound
                                self.best_value[i] = float(token)
                        
                        self.token_value[i] = self.best_value[i]   
                        
                        if self.episode-1 < len(self.token_range)*10: # first iterations of arms
                            index = int(self.episode / 10)
                            self.token_value[i] = self.token_range[index]
                        self.last_token_value[i] = self.token_value[i]       
                    self.epoch_returns = numpy.zeros(self.nr_agents, dtype=float) # reset
                        
            # AutoMATE        
            else:
                self.episode += 1
                for i in range(self.nr_agents):
                    self.epoch_values[i].append(self.values[i]/self.time_step)
                    # transition["values"][i].append(numpy.mean(self.epoch_values[i]))
                    self.values[i] = 0
                    if self.episode % 10 == 1:
                        # derive token value from value function
                        if self.episode > 10:
                            if len(self.rewards[i]) > 0:
                                self.min_reward[i] = abs(numpy.min(self.rewards[i]))
                                if numpy.max(self.rewards[i]) < 0: # If no positive rewards exist
                                    self.min_reward[i] = abs(numpy.max(self.rewards[i]))

                            # calculate s.t.e. gradient
                            if len(self.last_values[i]) > 0:
                                value_gradient = (numpy.median(self.epoch_values[i])-numpy.median(self.last_values[i]))/(numpy.median(self.last_values[i]))
                            else: # no update if no last s.t.e.
                                value_gradient = 0
                            # transition["value_gradients"][i].append(value_gradient)
                            
                            c = 0.1
                            update_rate = c * self.min_reward[i] 
                            
                            # if value change is too small
                            if abs(value_gradient) == numpy.inf:
                                value_gradient = 0.0 
                            
                            # update old token 
                            self.token_value[i] = self.token_value[i] + value_gradient * update_rate 
                            
                            # prevent negative token values
                            self.token_value[i] = numpy.maximum(0.0, self.token_value[i])
                            self.first_exchange[i] = True
                        
                        #reset episode parameters
                        self.last_values[i] = self.epoch_values[i]
                        self.epoch_values[i] = []

                # reset variables
                self.rewards = [[] for _ in range(self.nr_agents)] 
                self.last_rewards_observed = [[] for _ in range(self.nr_agents)]
                self.epoch_returns = numpy.zeros(self.nr_agents, dtype=float)
                        
            # track common token additionally in sovereign consensus 
            if self.no_sync:
                transition["token_values"][-1].append(self.common_token[i])
            self.time_step = 0
        return transition
