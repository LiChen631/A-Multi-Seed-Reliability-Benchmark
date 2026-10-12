"""
MATD3 Baseline Agent. Twin Q-networks with clipped double Q-learning.
"""
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from networks.actor import Actor, Critic
from utils.noise import OUNoise
from agents.maddpg import MADDPGAgent

class MATD3Agent(MADDPGAgent):
    def __init__(self, agent_id, obs_dim, global_state_dim, action_dim, n_agents,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005, hidden_dim=256,
                 target_noise=0.2, noise_clip=0.5, policy_delay=2, device='cpu'):
        super().__init__(agent_id, obs_dim, global_state_dim, action_dim, n_agents,
                         actor_lr, critic_lr, gamma, tau, hidden_dim, device)

        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.update_count = 0

        total_action_dim = action_dim * n_agents

        # Second critic (twin)
        self.critic2 = Critic(global_state_dim, total_action_dim, hidden_dim).to(device)
        self.target_critic2 = Critic(global_state_dim, total_action_dim, hidden_dim).to(device)
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer2 = Adam(self.critic2.parameters(), lr=critic_lr)

    def compute_target(self, next_global_state, next_obs, agent_actors, rewards_tensor, dones_tensor):
        """Clipped double Q-learning: use min(Q1, Q2) to reduce overestimation."""
        with torch.no_grad():
            target_actions = []
            for j in range(self.n_agents):
                nj = torch.FloatTensor(next_obs[j]).to(self.device)
                a_j = agent_actors[j](nj)
                noise = torch.randn_like(a_j) * self.target_noise
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                a_j = torch.clamp(a_j + noise, -1.0, 1.0)
                target_actions.append(a_j)
            target_actions_flat = torch.cat(target_actions, dim=1)

            gs = torch.FloatTensor(next_global_state).to(self.device)
            q1 = self.target_critic(gs, target_actions_flat)
            q2 = self.target_critic2(gs, target_actions_flat)
            q_min = torch.min(q1, q2)

            y = rewards_tensor[self.id].unsqueeze(1) + self.gamma * q_min * (1 - dones_tensor[self.id].unsqueeze(1))
        return y

    def update_critic(self, global_state, joint_actions, y_target):
        gs = torch.FloatTensor(global_state).to(self.device)
        ja = torch.FloatTensor(joint_actions).to(self.device)

        q1 = self.critic(gs, ja)
        loss1 = F.mse_loss(q1, y_target)
        self.critic_optimizer.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        q2 = self.critic2(gs, ja)
        loss2 = F.mse_loss(q2, y_target)
        self.critic_optimizer2.zero_grad()
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=10.0)
        self.critic_optimizer2.step()

        return (loss1.item() + loss2.item()) / 2

    def update_actor(self, obs_list, agent_actors, global_state_b=None):
        self.update_count += 1
        if self.update_count % self.policy_delay != 0:
            return 0.0
        return super().update_actor(obs_list, agent_actors, global_state_b)

    def soft_update(self):
        targets = [
            (self.target_actor, self.actor),
            (self.target_critic, self.critic),
            (self.target_critic2, self.critic2),
        ]
        for target, source in targets:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
