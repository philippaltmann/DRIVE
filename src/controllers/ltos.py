from src.utils import assertEquals, get_param_or_default
from src.controllers.actor_critic import ActorCritic, preprocessing_module
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy


class SharingNet(nn.Module):
    """High-level sharing policy phi_i(w_i^out | o_i).
    Returns a dense row of W with shape [batch, nr_agents].
    W[i, j] means: fraction of giver i's extrinsic return shared to recipient j.
    The row is normalized over allowed recipients only.
    """

    def __init__(self, agent_id, input_dim, nr_agents, nr_hidden_units, learning_rate, 
                 allowed_recipients=None, temperature=1.0, initial_own_share_ratio=None):
        super().__init__(); self.agent_id = agent_id; self.nr_agents = nr_agents; self.temperature = temperature
        if allowed_recipients is None: allowed_recipients = list(range(nr_agents))
        if agent_id not in allowed_recipients: allowed_recipients = list(allowed_recipients) + [agent_id]

        self.allowed_recipients = sorted(set(allowed_recipients))
        self.local_self_index = self.allowed_recipients.index(agent_id)

        self.fc_net = preprocessing_module(input_dim, nr_hidden_units)
        self.share_head = nn.Linear(nr_hidden_units, len(self.allowed_recipients))

        # Start from a controlled, stable sharing distribution, if initial_own_share_ratio is None, this is uniform.
        nn.init.zeros_(self.share_head.weight); nn.init.zeros_(self.share_head.bias)

        if initial_own_share_ratio is not None:
            n = len(self.allowed_recipients)
            if n > 1:
                own_ratio = min(max(float(initial_own_share_ratio), 1e-4), 1.0 - 1e-4)
                # Set logits so that softmax gives approximately:
                #   p(self) = own_ratio
                #   p(other) = (1 - own_ratio) / (n - 1)
                other_ratio = (1.0 - own_ratio) / float(n - 1)
                self_bias = numpy.log(own_ratio / other_ratio)
                with torch.no_grad(): self.share_head.bias[self.local_self_index] = float(self_bias)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, observations):
        batch_size = observations.size(0)
        x = observations.view(batch_size, -1)
        logits = self.share_head(self.fc_net(x))
        logits = logits / max(float(self.temperature), 1e-6)
        local_weights = F.softmax(logits, dim=-1)
        weights = torch.zeros(batch_size, self.nr_agents, dtype=local_weights.dtype, device=local_weights.device)
        weights[:, self.allowed_recipients] = local_weights
        return weights


class SharingConditionedActorNet(nn.Module):
    """Low-level actor pi_i(a_i | o_i, w_i^in)."""

    def __init__(self, input_dim, nr_agents, nr_actions, nr_hidden_units, learning_rate):
        super().__init__()
        self.nr_input_features = input_dim + nr_agents
        self.nr_hidden_units = nr_hidden_units
        self.fc_net = preprocessing_module(self.nr_input_features, nr_hidden_units)
        self.action_head = nn.Linear(nr_hidden_units, nr_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, observations, incoming_weights):
        batch_size = observations.size(0)
        obs = observations.view(batch_size, -1)
        w_in = incoming_weights.view(batch_size, -1)
        x = torch.cat([obs, w_in], dim=-1)
        x = self.fc_net(x)
        return F.softmax(self.action_head(x), dim=-1)


class SharingConditionedCriticNet(nn.Module):
    """Low-level critic V_i(o_i, w_i^in)."""

    def __init__(self, input_dim, nr_agents, nr_hidden_units, learning_rate):
        super().__init__()
        self.nr_input_features = input_dim + nr_agents
        self.nr_hidden_units = nr_hidden_units
        self.fc_net = preprocessing_module(self.nr_input_features, nr_hidden_units)
        self.value_head = nn.Linear(nr_hidden_units, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, observations, incoming_weights):
        batch_size = observations.size(0)
        obs = observations.view(batch_size, -1)
        w_in = incoming_weights.view(batch_size, -1)
        x = torch.cat([obs, w_in], dim=-1)
        x = self.fc_net(x)
        return self.value_head(x)


class LToS(ActorCritic):
    """Simple actor-critic-adapted Learning-to-Share controller according to 
    https://proceedings.neurips.cc/paper_files/paper/2022/file/61d8577984e4ef0cba20966eb3ef2ed8-Paper-Conference.pdf

    Optional params:
      - params["neighbors"]:
          list of lists. neighbors[i] contains agents that agent i may share with,
          including i. If omitted, sharing is fully connected.
      - params["share_temperature"]:
          softmax temperature for sharing weights.
      - params["initial_own_share_ratio"]:
          optional float in (0, 1). Initializes self-sharing bias, similar in spirit
          to SWARM's INITIAL_OWN_W_OUT_RATIO. If omitted, sharing starts uniform.
      - params["share_entropy_weight"]:
          optional entropy bonus for sharing policy. Default 0.0.
      - params["selfishness_reg"]:
          optional penalty on self-sharing. Default 0.0.
      - params["sharing_value_weight"]:
          multiplier for high-level value objective. Default 1.0.
    """

    def __init__(self, params):
        super().__init__(params)
        self.temperature = get_param_or_default(params, "share_temperature", 1.0)
        self.entropy_weight = get_param_or_default(params, "share_entropy_weight", 0.0)
        self.selfishness_reg = get_param_or_default(params, "selfishness_reg", 0.0)
        self.sharing_value_weight = get_param_or_default(params, "sharing_value_weight", 1.0)
        self.initial_own_share_ratio = get_param_or_default(params, "initial_own_share_ratio", None)

        # Replace the base actor/critic with sharing-conditioned variants.
        self.actor_nets = []
        self.critic_nets = []

        for _ in range(self.nr_agents):
            actor_net = SharingConditionedActorNet(
                input_dim=self.input_dim,
                nr_agents=self.nr_agents,
                nr_actions=self.nr_actions,
                nr_hidden_units=params["nr_hidden_units"],
                learning_rate=self.learning_rate,
            )
            self.actor_nets.append(actor_net.to(self.device))

            critic_net = SharingConditionedCriticNet(
                input_dim=self.input_dim,
                nr_agents=self.nr_agents,
                nr_hidden_units=params["nr_hidden_units"],
                learning_rate=self.learning_rate,
            )
            self.critic_nets.append(critic_net.to(self.device))

        neighbors = get_param_or_default(params, "neighbors", None)

        self.sharing_nets = []
        for i in range(self.nr_agents):
            allowed = None if neighbors is None else neighbors[i]

            sharing_net = SharingNet(
                agent_id=i,
                input_dim=self.input_dim,
                nr_agents=self.nr_agents,
                nr_hidden_units=params["nr_hidden_units"],
                learning_rate=self.learning_rate,
                allowed_recipients=allowed,
                temperature=self.temperature,
                initial_own_share_ratio=self.initial_own_share_ratio,
            )
            self.sharing_nets.append(sharing_net.to(self.device))

        self.token_values = [0.0 for _ in range(self.nr_agents)]

        # Similar in spirit to alternating low-level/high-level training.
        # If this is too slow, set params["update_sharing_every_step"] = True.
        self.update_sharing_every_step = get_param_or_default(params, "update_sharing_every_step", False)
        self.update_sharing = True

    def update_step(self):
        preprocessed_data = self.preprocess()
        if self.update_sharing or self.update_sharing_every_step:
            for i, memory, sharing_net in zip(range(self.nr_agents), self.memories, self.sharing_nets):
                self.local_sharing_update(i, memory, sharing_net, preprocessed_data)
        for i, memory, actor_net, critic_net in zip(range(self.nr_agents), self.memories, self.actor_nets, self.critic_nets):
            self.local_update(i, memory, actor_net, critic_net, preprocessed_data)
        for memory in self.memories: memory.clear()
        self.update_sharing = not self.update_sharing

    def local_probs(self, history, agent_id):
        """Fallback action probabilities for the existing controller API. The original SWARM action policy receives current w_in. 
        The current codebase's local_probs() API only passes one agent's history, so we use a neutral uniform incoming vector for 
        online action selection. During training, the actor and critic are conditioned on the actual W-derived incoming weights."""
        history = torch.tensor(numpy.array([history]), dtype=torch.float32, device=self.device)
        uniform_w_in = torch.full((1, self.nr_agents), 1.0 / float(self.nr_agents), dtype=torch.float32, device=self.device)
        return self.actor_nets[agent_id](history, uniform_w_in).detach().cpu().numpy()[0]

    def _collect_histories_and_returns(self):
        histories = []; extrinsic_returns = []
        for memory in self.memories:
            h, _, _, _, returns, _, _, _ = memory.get_training_data()
            histories.append(h.to(self.device))
            extrinsic_returns.append(returns.to(self.device))
        returns_matrix = torch.stack(extrinsic_returns, dim=1).detach()
        return histories, returns_matrix

    def _compute_W(self, histories, detach=True):
        rows = []
        for h, sharing_net in zip(histories, self.sharing_nets):
            row = sharing_net(h)
            if detach: row = row.detach()
            rows.append(row)
        W = torch.stack(rows, dim=1)
        return W

    def _incoming_from_W(self, W, recipient_id):
        """Return w_recipient^in. W shape: [T, giver, recipient]
        Incoming weights for recipient j: W[:, :, j], shape [T, nr_agents] """
        return W[:, :, recipient_id]

    def preprocess(self):
        histories, returns_matrix = self._collect_histories_and_returns()
        T = returns_matrix.size(0)
        with torch.no_grad():
            W = self._compute_W(histories, detach=True)
            assertEquals(W.size(), torch.Size([T, self.nr_agents, self.nr_agents]))
            # LToS reward sharing: shaped_return_j(t) = sum_i W[t, i, j] * env_return_i(t)
            shaped_returns = (W * returns_matrix[:, :, None]).sum(dim=1)
            incoming_weights = [self._incoming_from_W(W, j).detach() for j in range(self.nr_agents)]
            for i in range(self.nr_agents): self.token_values[i] = W[:, i, i].mean().item()
        return (W.detach(), returns_matrix.detach(), shaped_returns.detach(), incoming_weights, histories)

    def update_critic(self, agent_id, training_data, critic_net, preprocessed_data):
        _, _, shaped_returns, incoming_weights, _ = preprocessed_data
        histories, _, _, _, returns, _, _, _ = training_data
        histories = histories.to(self.device)
        w_in = incoming_weights[agent_id].to(self.device)
        target_returns = shaped_returns[:, agent_id].to(self.device)
        assertEquals(returns.size(), target_returns.size())
        values = critic_net(histories, w_in).squeeze()
        assertEquals(values.size(), target_returns.size())
        critic_loss = F.mse_loss(target_returns.detach(), values)
        critic_net.optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic_net.parameters(), self.clip_norm)
        critic_net.optimizer.step()

    def get_values(self, agent_id, histories, incoming_weights=None):
        if incoming_weights is None: incoming_weights = torch.full((histories.size(0), self.nr_agents), 1.0 / float(self.nr_agents), dtype=torch.float32, device=self.device)
        return self.critic_nets[agent_id](histories.to(self.device), incoming_weights.to(self.device))

    def update_actor(self, agent_id, training_data, actor_net, preprocessed_data):
        _, _, shaped_returns, incoming_weights, _ = preprocessed_data
        histories, _, actions, _, returns, old_probs, _, _ = training_data
        histories = histories.to(self.device)
        actions = actions.to(self.device)
        old_probs = old_probs.to(self.device)
        w_in = incoming_weights[agent_id].to(self.device)
        target_returns = shaped_returns[:, agent_id].to(self.device)
        assertEquals(returns.size(), target_returns.size())
        values = self.get_values(agent_id, histories, w_in).squeeze().detach()
        action_probs = actor_net(histories, w_in)
        advantages = target_returns.detach() - values.detach()
        actor_loss = self.policy_loss(advantages.detach(), action_probs, actions, old_probs).sum()
        actor_net.optimizer.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(actor_net.parameters(), self.clip_norm)
        actor_net.optimizer.step()

    def policy_loss(self, advantage, probs, action, old_probs):
        # Keep the base actor-critic behavior.
        return -Categorical(probs).log_prob(action) * advantage

    def _set_critic_requires_grad(self, requires_grad):
        [p.requires_grad_(requires_grad) for critic_net in self.critic_nets for p in critic_net.parameters()]

    def local_sharing_update(self, agent_id, memory, sharing_net, preprocessed_data):
        """Update one sharing row using critic-gradient signal."""
        W_detached, _, _, _, histories = preprocessed_data

        # Recompute this agent's outgoing row with a fresh graph.
        histories_i = histories[agent_id].to(self.device)
        row_i = sharing_net(histories_i); rows = []
        for k in range(self.nr_agents):
            if k == agent_id: rows.append(row_i)
            else: rows.append(W_detached[:, k, :].detach())

        W_candidate = torch.stack(rows, dim=1)

        # During the sharing update, critics provide gradients wrt w_in but their own parameters should not be updated.
        self._set_critic_requires_grad(False); global_values = []
        for recipient_id in range(self.nr_agents):
            h_j = histories[recipient_id].to(self.device)
            w_in_j = self._incoming_from_W(W_candidate, recipient_id)
            v_j = self.critic_nets[recipient_id](h_j, w_in_j).squeeze()
            global_values.append(v_j)

        global_value = torch.stack(global_values, dim=1).sum(dim=1).mean()
        value_loss = -float(self.sharing_value_weight) * global_value
        weights_safe = row_i.clamp_min(1e-8)
        entropy = -(weights_safe * weights_safe.log()).sum(dim=1).mean()
        entropy_loss = -float(self.entropy_weight) * entropy
        selfishness_loss = float(self.selfishness_reg) * row_i[:, agent_id].mean()
        loss = value_loss + entropy_loss + selfishness_loss
        sharing_net.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(sharing_net.parameters(), self.clip_norm)
        sharing_net.optimizer.step()
        self._set_critic_requires_grad(True)
