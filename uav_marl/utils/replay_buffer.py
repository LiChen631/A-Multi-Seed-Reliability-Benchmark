"""Experience Replay Buffer for multi-agent DRL."""
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, actions, rewards, next_obs, global_state, next_global_state, done):
        self.buffer.append((obs, actions, rewards, next_obs, global_state, next_global_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, gs, ngs, dones = zip(*batch)
        # Stack into batched arrays
        obs_b = {i: np.stack([o[i] for o in obs]) for i in range(len(obs[0]))}
        act_b = {i: np.stack([a[i] for a in actions]) for i in range(len(actions[0]))}
        rew_b = {i: np.stack([r[i] for r in rewards]) for i in range(len(rewards[0]))}
        nobs_b = {i: np.stack([n[i] for n in next_obs]) for i in range(len(next_obs[0]))}
        gs_b = np.stack(gs)
        ngs_b = np.stack(ngs)
        done_b = {i: np.stack([d[i] for d in dones]) for i in range(len(dones[0]))}
        return obs_b, act_b, rew_b, nobs_b, gs_b, ngs_b, done_b

    def __len__(self):
        return len(self.buffer)
