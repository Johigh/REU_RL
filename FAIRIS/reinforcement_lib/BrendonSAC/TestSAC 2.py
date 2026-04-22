import os
import sys
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import time

# Path for FAIRIS
sys.path.append(r"c:\Users\Johan\OneDrive\Documents\School\FAIRIS")
os.chdir(r"c:\Users\Johan\OneDrive\Documents\School\FAIRIS")

import numpy as np
import torch
#from reinforcement_lib.PPO.PPONetworks import Agent
from reinforcement_lib.SAC.Agent import Agent
from fairis_tools.experiment_tools.loggers.experiment_logger import ExperimentLogger, EpisodeLogger
from reinforcement_lib.BrendonSAC.FAIRISEnvNoPC import FAIRISEnvTF
from fairis_tools.experiment_tools.paths.path_planning.MazePaths import MazePaths

#********** Gets the shortest path for a maze*************
def get_shortest_path(filename):
    maze_path = MazePaths(filename)
    maze_path.calculate_shortest_paths()
    return len(maze_path.paths[0].path)

# Used to visualize the algorithm during a run by draw its path during an episode
def draw_path(path, filename, walls, borders):
    plt.clf()
    fig, ax = plt.subplots()
    rect = patches.Rectangle(borders[0], borders[1][0], borders[1][1], linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.relim()
    ax.autoscale_view()

#    for wall in walls:
#        ax.plot(wall[0], wall[1], 'black')

    # Plot path
    data = np.asarray(path)
    ax.plot(data[:, 0], data[:, 1], 'g')

    # Save file
    plt.savefig(filename)
    plt.close()

    
maze_name = "fourrooms"
print(f"Maze name is {maze_name}")
maze_file = r'c:\Users\Johan\OneDrive\Documents\School\FAIRIS\simulation\worlds\mazes\Experiment1\WM10.xml'

# Create data directory if it doesn't exist
data_dir = r'c:\Users\Johan\OneDrive\Documents\School\FAIRIS\data'
os.makedirs(data_dir, exist_ok=True)

# Create environment wrapper for FAIRISEnvNoPC
class EnvWrapper:
    """Wrapper to make FAIRISEnvNoPC compatible with existing code"""
    def __init__(self, env):
        self.env = env
        self.action_space = type('obj', (object,), {'n': 8})()  # 8 actions
        self._observation_shape = (3,)  # x, y coordinates, and lidar reading
        
    def observation_spec(self):
        return type('obj', (object,), {'shape': self._observation_shape})()
    
    def reset(self):
        return np.array(self.env.reset(), dtype=np.float32)
    
    def step(self, action):
        state, reward, done = self.env.step(action)
        # Convert state tuple to array and add info with robot position
        state_array = np.array(state, dtype=np.float32)
        info = [state[0], state[1], state[2]]  # x, y, lidar for info
        return state_array, reward, done, info
    
    def get_robot_pose(self):
        return self.env.robot.get_robot_pose()
    
    def reset_environment(self):
        self.env.first_run = True

env_base = FAIRISEnvTF(
    maze_filet=maze_file,
    horizont=250,  # max_steps_per_episode equivalent
    sequence_learningt=False,  # not using sequence learning
    reward_lent=0  # reward length for subgoal completion
)
env = EnvWrapper(env_base)
load_checkpoint = False
ver_name = '_'+maze_file+'_rewardV3'
# gamma = 0.9
N = 5#20
batch_size = 32 #10
alpha = 1e-3 #changed alpha to 1e-3
experiment_logger = ExperimentLogger(maze_file,'uniform_32')
max_steps = 175000 

n_layers_vals =  2 #[2, 3, 4]
layer_sizes = 64 #[32, 64, 128, 256]
p_data = {}
# idy = 3
# idx = 0 # new change - single configuration

print(f"Starting run: {n_layers_vals}, {layer_sizes}\n\n")

agent = Agent(n_actions=env.action_space.n,
                          batch_size=batch_size,
                          alpha=alpha,
                          input_dims=env.observation_spec().shape,
                          ver_name=ver_name,
                          n_layers=n_layers_vals,
                          layer_size=layer_sizes)

n_episodes = 5#2500
score_history = []
avg_score = 0
learn_iters = 0
n_steps = 0
lengths = []
steps = []
shortest_path = []
min_len = 5000

#added for new log info 
all_episodes_logs = []
log_file = os.path.join(data_dir, f"{maze_name}_debug")

#for i in range(n_episodes):
while n_steps < max_steps:
    observation = env.reset()
    time.sleep(0.05)
    path_length = 0
    done = False
    score = 0
    start_x, start_y, start_theta = env.get_robot_pose()
    episode_logger = EpisodeLogger(start_x, start_y)
    prev_action = None
    path = [(start_x, start_y)]
    ep_actions = [] #added for new log info 
    ep_states_x = [] # - 
    ep_states_y = [] # -
    ep_states_lidar = []
    ep_rewards = []  # -
    ep_metrics = [] #  -
    losses = []
    while not done:
        action = agent.choose_action(observation)
        prev_action = action
        observation_, reward, done, info = env.step(action)
        n_steps += 1
        episode_logger.log_step(robot_x=info[0],robot_y=info[1])
        path.append((info[0], info[1]))
        path_length += 1
        score += reward
        # new info for logger
        ep_actions.append(action)
        ep_states_x.append(info[0])
        ep_states_y.append(info[1])
        ep_states_lidar.append(info[2])
        ep_rewards.append(reward)

        agent.remember(observation, action, reward, observation_, done)
        if n_steps % N == 0:
            metrics = agent.learn()
            if metrics is not None:
                ep_metrics.append(metrics)
            learn_iters += 1
        observation = observation_

    experiment_logger.log_episode(episode_logger,score,path_length)
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    lengths.append(path_length)
    steps.append(n_steps)

    #Compute EPISODE DEBUG stats
    ep_actions_arr = np.array(ep_actions)
    ep_states_x_arr = np.array(ep_states_x)
    ep_states_y_arr = np.array(ep_states_y)
    ep_states_lidar_arr = np.array(ep_states_lidar)
    ep_rewards_arr = np.array(ep_rewards)

    action_counts = np.bincount(ep_actions_arr, minlength = 8)
    action_pcts = action_counts / len(ep_actions_arr) * 100

    if ep_metrics: 
        def avg_metrics(key):
            return np.mean([m[key] for m in ep_metrics])
        metric_summary = {k: avg_metrics(k) for k in ep_metrics[0].keys()}
    else:
        metric_summary = {}


    episode_log = {
        'episode': len(all_episodes_logs) + 1,
        'n_steps': n_steps,
        'path_length': path_length,
        'score': score,
        'avg_score_100': avg_score,
        'actions': {
            'mean': ep_actions_arr.mean(),
            'std': ep_actions_arr.std(),
            'min': int(ep_actions_arr.min()),
            'max': int(ep_actions_arr.max()),
            'distribution_pct': action_pcts.tolist(),
        },
        'states': {
            'x_mean': ep_states_x_arr.mean(), 'x_std': ep_states_x_arr.std(),
            'x_min': ep_states_x_arr.min(), 'x_max': ep_states_x_arr.max(),
            'y_mean': ep_states_y_arr.mean(), 'y_std': ep_states_y_arr.std(),
            'y_min': ep_states_y_arr.min(), 'y_max': ep_states_y_arr.max(),
            'lidar_mean': ep_states_lidar_arr.mean(), 'lidar_std': ep_states_lidar_arr.std(),
            'lidar_min': ep_states_lidar_arr.min(), 'lidar_max': ep_states_lidar_arr.max(),
        },
        'rewards': {
            'mean': ep_rewards_arr.mean(), 'std': ep_rewards_arr.std(),
            'min': ep_rewards_arr.min(), 'max': ep_rewards_arr.max(),
        },
        'metrics': metric_summary,
    }

    all_episodes_logs.append(episode_log)
    with open(log_file + 'debug_log.pkl', 'wb') as f:
        pickle.dump(all_episodes_logs, f)

     # Console output
    print(f"\n{'='*80}")
    print(f"EP {episode_log['episode']} | {datetime.datetime.now().strftime('%H:%M:%S')} | steps: {n_steps} | path_len: {path_length} | score: {score:.2f} | avg: {avg_score:.2f}")
    print(f"  Actions  -> mean: {ep_actions_arr.mean():.1f} std: {ep_actions_arr.std():.1f} | dist: {['%.0f%%'%p for p in action_pcts]}")
    print(f"  States   -> x:[{ep_states_x_arr.min():.2f}, {ep_states_x_arr.max():.2f}] y:[{ep_states_y_arr.min():.2f}, {ep_states_y_arr.max():.2f}] lidar:[{ep_states_lidar_arr.min():.2f}, {ep_states_lidar_arr.max():.2f}]")
    print(f"  Rewards  -> mean: {ep_rewards_arr.mean():.3f} std: {ep_rewards_arr.std():.3f} min: {ep_rewards_arr.min():.2f} max: {ep_rewards_arr.max():.2f}")
    if metric_summary:
        print(f"  Losses   -> val: {metric_summary['value_loss']:.4f} actor: {metric_summary['actor_loss']:.4f} c1: {metric_summary['critic_1_loss']:.4f} c2: {metric_summary['critic_2_loss']:.4f}")
        print(f"  Q-values -> q1[{metric_summary['q1_min']:.2f}, {metric_summary['q1_mean']:.2f}, {metric_summary['q1_max']:.2f}] q2[{metric_summary['q2_min']:.2f}, {metric_summary['q2_mean']:.2f}, {metric_summary['q2_max']:.2f}]")
        print(f"  Grads    -> actor: {metric_summary['actor_grad_norm']:.4f} value: {metric_summary['value_grad_norm']:.4f} c1: {metric_summary['critic1_grad_norm']:.4f} c2: {metric_summary['critic2_grad_norm']:.4f}")
        print(f"  Entropy  -> {metric_summary['entropy']:.4f}")
    print(f"{'='*80}")

    converged = len(lengths) >= 20 and max(lengths[-20:]) < 60
    if converged:
        finished = True
        print(f"Converge time: {n_steps}")
        break

    if path_length < min_len:
        shortest_path = path
        min_len = path_length

    # print(f"{datetime.datetime.now()}: step: {n_steps}, path len: {path_length}, learn iters: {learn_iters}")

print(f"Finished conv time: {n_steps}")
print(min(lengths))

p_data[f"{n_layers_vals}-{layer_sizes}"] = {"lens": lengths, "steps": steps}

env.reset_environment()

with open(fr"c:\Users\Johan\OneDrive\Documents\School\FAIRIS\data\{maze_name}_param_data.pkl", 'wb') as fp:
    pickle.dump(p_data, fp)