"""
MADDPG Baseline Agent. Single Q-network per agent (overestimation bias).
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils
from torch.optim import Adam
from networks.actor import Actor, Critic
from utils.noise import OUNoise

class MADDPGAgent:
    def __init__(self, agent_id, obs_dim, global_state_dim, action_dim, n_agents,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005, hidden_dim=256, device='cpu'):
        self.id = agent_id
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device

        total_action_dim = action_dim * n_agents

        # Networks
        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.target_actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = Critic(global_state_dim, total_action_dim, hidden_dim).to(device)
        self.target_critic = Critic(global_state_dim, total_action_dim, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        self.noise = OUNoise(action_dim)

    def select_action(self, obs, add_noise=True):
        with torch.no_grad():
            o = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            a = self.actor(o).cpu().numpy().flatten()
        if add_noise:
            a += self.noise.sample()
        return np.clip(a, -1.0, 1.0)

    def compute_target(self, next_global_state, next_obs, agent_actors, rewards_tensor, dones_tensor):
        """Compute target Q-value for critic update."""
        target_actions = []
        for j in range(self.n_agents):
            with torch.no_grad():
                nj = torch.FloatTensor(next_obs[j]).to(self.device)
                a_j = agent_actors[j](nj)
                target_actions.append(a_j)
        target_actions_flat = torch.cat(target_actions, dim=1)

        with torch.no_grad():
            gs = torch.FloatTensor(next_global_state).to(self.device)
            q_target = self.target_critic(gs, target_actions_flat)
            y = rewards_tensor[self.id].unsqueeze(1) + self.gamma * q_target * (1 - dones_tensor[self.id].unsqueeze(1))
        return y

    def update_critic(self, global_state, joint_actions, y_target):
        gs = torch.FloatTensor(global_state).to(self.device)
        ja = torch.FloatTensor(joint_actions).to(self.device)
        q_pred = self.critic(gs, ja)
        loss = F.mse_loss(q_pred, y_target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()
        return loss.item()

    def update_actor(self, obs_list, agent_actors, global_state_b):
        o_i = torch.FloatTensor(obs_list[self.id]).to(self.device)
        a_i = self.actor(o_i)
        all_actions = []
        for j in range(self.n_agents):
            if j == self.id:
                all_actions.append(a_i)
            else:
                with torch.no_grad():
                    o_j = torch.FloatTensor(obs_list[j]).to(self.device)
                    all_actions.append(agent_actors[j](o_j))
        all_actions_flat = torch.cat(all_actions, dim=1)
        gs = torch.FloatTensor(global_state_b).to(self.device)
        loss = -self.critic(gs, all_actions_flat).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss.item()

    def soft_update(self):
        for target, source in [(self.target_actor, self.actor),
                               (self.target_critic, self.critic)]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def reset_noise(self):
        self.noise.reset()
