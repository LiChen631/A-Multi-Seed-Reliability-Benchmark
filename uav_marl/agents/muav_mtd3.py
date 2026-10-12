"""
MUAV-MTD3: Multi-UAV Mutual Twin Delayed DDPG (Proposed Method 1).
Target: y = beta*min(Q1,Q2) + (1-beta)*mean(peer min(Q1,Q2) estimates)
"""
import torch
import torch.nn.functional as F
from agents.matd3 import MATD3Agent

class MUAVMTD3Agent(MATD3Agent):
    def __init__(self, agent_id, obs_dim, global_state_dim, action_dim, n_agents,
                 beta_mutual=0.5, **kwargs):
        super().__init__(agent_id, obs_dim, global_state_dim, action_dim, n_agents, **kwargs)
        self.beta_mutual = beta_mutual

    def compute_target(self, next_global_state, next_obs, agent_actors, rewards_tensor, dones_tensor,
                       peer_critics=None):
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
            y_min = torch.min(q1, q2)

            if peer_critics and len(peer_critics) > 0:
                peer_qs = []
                for pq1, pq2 in peer_critics:
                    pq = torch.min(pq1(gs, target_actions_flat), pq2(gs, target_actions_flat))
                    peer_qs.append(pq)
                y_mutual = torch.stack(peer_qs).mean(dim=0)
            else:
                y_mutual = y_min

            y_q = self.beta_mutual * y_min + (1 - self.beta_mutual) * y_mutual
            y = rewards_tensor[self.id].unsqueeze(1) + self.gamma * y_q * (1 - dones_tensor[self.id].unsqueeze(1))
        return y
