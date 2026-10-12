"""
MUAV-STD3: Multi-UAV Softmax Twin Delayed DDPG (Proposed Method 2).
Target: softmax over K sampled actions, applied to max(Q1, Q2).
"""
import torch
import torch.nn.functional as F
from agents.matd3 import MATD3Agent

class MUAVSTD3Agent(MATD3Agent):
    def __init__(self, agent_id, obs_dim, global_state_dim, action_dim, n_agents,
                 beta_soft=1.0, k_samples=10, **kwargs):
        super().__init__(agent_id, obs_dim, global_state_dim, action_dim, n_agents, **kwargs)
        self.beta_soft = beta_soft
        self.k_samples = k_samples

    def compute_target(self, next_global_state, next_obs, agent_actors, rewards_tensor, dones_tensor):
        with torch.no_grad():
            # Other agents' target actions (fixed)
            fixed_actions = []
            for j in range(self.n_agents):
                nj = torch.FloatTensor(next_obs[j]).to(self.device)
                a_j = agent_actors[j](nj)
                fixed_actions.append(a_j)

            gs = torch.FloatTensor(next_global_state).to(self.device)
            a_i_center = fixed_actions[self.id]

            all_sampled_qs = []
            for k in range(self.k_samples):
                noise = torch.randn_like(a_i_center) * self.target_noise
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                a_i_k = torch.clamp(a_i_center + noise, -1.0, 1.0)

                all_actions = []
                for j in range(self.n_agents):
                    if j == self.id:
                        all_actions.append(a_i_k)
                    else:
                        all_actions.append(fixed_actions[j])
                actions_flat = torch.cat(all_actions, dim=1)

                q1 = self.target_critic(gs, actions_flat)
                q2 = self.target_critic2(gs, actions_flat)
                all_sampled_qs.append(torch.max(q1, q2))

            qs_stacked = torch.stack(all_sampled_qs, dim=0)  # (K, B, 1)
            qs_stacked = qs_stacked.squeeze(-1).T  # (B, K)

            weights = F.softmax(self.beta_soft * qs_stacked, dim=-1)
            expected_q = (weights * qs_stacked).sum(dim=-1, keepdim=True)

            y = rewards_tensor[self.id].unsqueeze(1) + self.gamma * expected_q * (1 - dones_tensor[self.id].unsqueeze(1))
        return y
