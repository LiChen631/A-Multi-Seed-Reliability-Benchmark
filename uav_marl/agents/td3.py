"""
TD3 Baseline Agent (single-agent, independent training per UAV).
Standard TD3: twin critics + clipped double Q-learning + target policy smoothing.
Each UAV trains its own actor/critic using only its local observation.
"""
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from networks.actor import Actor, Critic


class TD3Agent:
    def __init__(self, agent_id, obs_dim, action_dim,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005, hidden_dim=256,
                 target_noise=0.2, noise_clip=0.5, policy_delay=2, device='cpu'):
        self.id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.update_count = 0

        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.target_actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Critic takes (obs, action) — single-agent
        self.critic = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.target_critic = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic2 = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.target_critic2 = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_optimizer2 = Adam(self.critic2.parameters(), lr=critic_lr)

    def select_action(self, obs, add_noise=True):
        o = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a = self.actor(o).cpu().numpy().squeeze()
        if add_noise:
            a = a + np.random.randn(self.action_dim) * 0.1
        return np.clip(a, -1.0, 1.0)

    def reset_noise(self):
        pass

    def compute_target(self, next_obs, rewards, dones):
        """r + gamma * min(Q1', Q2')(s', a' + clipped noise)"""
        with torch.no_grad():
            no = torch.FloatTensor(next_obs).to(self.device)
            a_next = self.target_actor(no)
            noise = torch.randn_like(a_next) * self.target_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            a_next = torch.clamp(a_next + noise, -1.0, 1.0)

            q1 = self.target_critic(no, a_next)
            q2 = self.target_critic2(no, a_next)
            q_min = torch.min(q1, q2)
            rew = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            d = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
            y = rew + self.gamma * q_min * (1 - d)
        return y

    def update_critic(self, obs, actions, y_target):
        o = torch.FloatTensor(obs).to(self.device)
        a = torch.FloatTensor(actions).to(self.device)
        q1 = self.critic(o, a)
        loss1 = F.mse_loss(q1, y_target)
        self.critic_optimizer.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()
        q2 = self.critic2(o, a)
        loss2 = F.mse_loss(q2, y_target)
        self.critic_optimizer2.zero_grad()
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=10.0)
        self.critic_optimizer2.step()
        return (loss1.item() + loss2.item()) / 2

    def update_actor(self, obs):
        self.update_count += 1
        if self.update_count % self.policy_delay != 0:
            return 0.0
        o = torch.FloatTensor(obs).to(self.device)
        a = self.actor(o)
        q = self.critic(o, a)
        loss = -q.mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()
        return loss.item()

    def soft_update(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
