"""Actor (policy) and Critic (Q) networks."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Deterministic policy: local obs -> action."""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs)

class Critic(nn.Module):
    """Centralized Q-function: [global_state, global_actions] -> Q-value."""
    def __init__(self, global_state_dim, total_action_dim, hidden_dim=256):
        super().__init__()
        input_dim = global_state_dim + total_action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=-1)
        return self.net(x)
