"""Complete MUAV-STD3 seed30 only."""
import sys, os, json, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(r"C:\Users\Administrator\Desktop\2222\uav_marl")

import numpy as np, torch
from config import *
from env.uav_env import UAVFormationEnv
from utils.replay_buffer import ReplayBuffer
from train import build_env, create_agents, update_agents

BASE = os.environ.get('UAV_RESULTS_DIR', 'results')
torch.manual_seed(30); np.random.seed(30)
EPISODES = 800

sd = os.path.join(BASE, 'checkpoints', 'muav_std3_seed30_v2')
if os.path.exists(sd): shutil.rmtree(sd)
os.makedirs(sd)

env = build_env(); agents = create_agents('muav_std3', env)
ep_rewards = []; buffer = ReplayBuffer(REPLAY_SIZE); total_steps = 0

for ep in range(1, EPISODES + 1):
    obs_list = env.reset(); gs = env._get_global_state()
    for a in agents: a.reset_noise()
    ep_r = np.zeros(N_AGENTS)
    for step in range(MAX_EPISODE_STEPS):
        actions = [agents[i].select_action(obs_list[i], add_noise=True) for i in range(N_AGENTS)]
        nobs, rewards, dones, info = env.step(actions)
        buffer.push(obs_list, actions, rewards, nobs, gs, env._get_global_state(), np.array([1.0 if dones[0] else 0.0]*N_AGENTS, dtype=np.float32))
        ep_r += np.array(rewards); obs_list = nobs; gs = env._get_global_state(); total_steps += 1
        if total_steps >= WARMUP_STEPS and len(buffer) >= BATCH_SIZE: update_agents(agents, buffer, 'muav_std3')
        if dones[0]: break
    ep_rewards.append(ep_r.sum())
    if ep % 200 == 0 or ep == 1:
        print(f"  Ep{ep:4d} Avg100:{np.mean(ep_rewards[-min(100,len(ep_rewards)):]):8.1f}")

for i, ag in enumerate(agents):
    state = {'actor': ag.actor.state_dict(), 'critic': ag.critic.state_dict(), 'target_actor': ag.target_actor.state_dict(), 'target_critic': ag.target_critic.state_dict()}
    if hasattr(ag,'critic2'): state['critic2'] = ag.critic2.state_dict()
    torch.save(state, os.path.join(sd, f'agent_{i}.pt'))
torch.save({'episode': EPISODES, 'rewards': ep_rewards, 'seed': 30}, os.path.join(sd, 'train_state.pt'))
res = {'algo': 'muav_std3', 'seed': 30, 'best': float(max(ep_rewards)), 'final100': float(np.mean(ep_rewards[-100:]))}
with open(os.path.join(BASE, 'muav_std3_seed30_v2.json'), 'w') as f: json.dump(res, f)
print(f"DONE: best={res['best']:.1f} final100={res['final100']:.1f}")
