"""
Master script: wait for training, then run all experiments.
Usage: python run_all_experiments.py
"""
import subprocess, sys, os, time, json

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
BASE_DIR = r"C:\Users\Administrator\Desktop\2222\uav_marl"

ALGOS = ['matd3', 'muav_mtd3', 'muav_std3']  # maddpg already trained

def is_trained(algo):
    ckpt = os.path.join(RESULTS_DIR, 'checkpoints', algo, 'train_state.pt')
    if not os.path.exists(ckpt):
        return False
    import torch
    ts = torch.load(ckpt, map_location='cpu', weights_only=False)
    return ts.get('completed', False)

def train(algo, episodes=800):
    print(f"\n{'='*50}")
    print(f"Training {algo} ({episodes} episodes)")
    print(f"{'='*50}")
    result = subprocess.run(
        [sys.executable, 'train.py', '--algo', algo, '--episodes', str(episodes)],
        cwd=BASE_DIR, capture_output=False
    )
    return result.returncode == 0

def run_experiment(task):
    print(f"\n{'='*50}")
    print(f"Running experiment: {task}")
    print(f"{'='*50}")
    result = subprocess.run(
        [sys.executable, 'experiments.py', '--task', task],
        cwd=BASE_DIR, capture_output=False
    )
    return result.returncode == 0

if __name__ == '__main__':
    # Train missing algorithms
    for algo in ALGOS:
        if is_trained(algo):
            print(f"[{algo}] Already trained, skipping")
        else:
            train(algo, episodes=800)

    print("\n" + "="*60)
    print("All algorithms trained. Running experiments...")
    print("="*60)

    # Run all experiments
    tasks = ['qbias', 'hyperparam', 'robustness', 'trajectory']
    for task in tasks:
        run_experiment(task)

    print("\n" + "="*60)
    print("ALL DONE! Check results/figures/ for outputs.")
    print("="*60)
