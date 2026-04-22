import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Path for FAIRIS
sys.path.append(r"c:\Users\Johan\OneDrive\Documents\School\FAIRIS")
os.chdir(r"c:\Users\Johan\OneDrive\Documents\School\FAIRIS")

import torch
import torch.nn as nn
import numpy as np
from collections.abc import Callable
from numbers import Number
from torch.utils.tensorboard import SummaryWriter
import datetime

from tianshou.algorithm import SAC
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.sac import SACPolicy#, DiscreteSAC
from tianshou.algorithm.modelfree.sac import AutoAlpha
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, ReplayBuffer #VectorReplayBuffer
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.continuous import ContinuousCritic, ContinuousActorProbabilistic#DiscreteActor, DiscreteCritic
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils import BaseLogger, LazyLogger
from tianshou.utils import TensorboardLogger

from FAIRIS_Env_Gym_continous import FAIRISEnv

# Hyperparameters
hidden_size = 256
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
tau= 0.005
gamma= 0.99
n_step= 3
frames_stack= 1
buffer_size= 100000
epoch= 100
epoch_num_steps= 1000 #decrease for "new" test maze 4/13
collection_step_num_env_steps = 100
batch_size=256
update_per_step=1.0
hidden_sizes = [256, 256]
save_weights = True
weight_path = f"./sac_weights.pth"

# Env parameters
maze_file = r'C:\\Users\\Johan\\OneDrive\\Documents\\School\\FAIRIS\\simulation\\worlds\\mazes\\Experiment1\\WM10.xml'
horizon = 500

VALID_LOG_VALS_TYPE = int | Number | np.number | np.ndarray | float

# Simple Logger to printto screen or file
class EpisodeLengthLogger(BaseLogger):
    def __init__(self):
        super().__init__(training_interval=500)
        self.data = []
        self.printFile = False
        if self.printFile:
            self.file_name = r"/home/brendon/PhDResearch/REU/FAIRIS/reinforcement_lib/torchrl/log.txt"
            with open(self.file_name, 'w') as file:
                pass

    def prepare_dict_for_logging(
        self,
        data: dict[str, VALID_LOG_VALS_TYPE],
    ) -> dict[str, VALID_LOG_VALS_TYPE]:
        return data

    def write(self, step_type: str, step: int, data: dict[str, VALID_LOG_VALS_TYPE]) -> None:
        """The LazyLogger writes nothing."""
        if ('lens' in data.keys()) and ('returns' in data.keys()) and (len(data['lens']) > 0):
            if self.printFile:
                with open(self.file_name, 'a') as file:
                    file.write(f"Step: {step}, lens: {data['lens']}, returns: {data['returns']}\n")
            else:
                print(f"Step: {step}, lens: {data['lens']}, returns: {data['returns']}, collected_eps: {data['n_collected_episodes']}")

    def finalize(self) -> None:
        pass

    def save_data(
        self,
        epoch: int,
        env_step: int,
        update_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        pass

    def restore_data(self) -> tuple[int, int, int]:
        return 0, 0, 0

    @staticmethod
    def restore_logged_data(log_path: str) -> dict:
        return {}



# Build Env
# Create Function to make the env to be passed to the DummyVectorEnv
def create_env():
    return FAIRISEnv(maze_filet=maze_file, horizont=horizon, continuous=True)
vector_env = DummyVectorEnv([create_env])

# Action shape and Obs shape
action_shape = vector_env.action_space[0].shape
obs_shape = vector_env.observation_space[0].shape

# Define networks used for actor and both critics
a_model = Net(state_shape=obs_shape[0], hidden_sizes=hidden_sizes)
c1_model = Net(state_shape=obs_shape[0], hidden_sizes=hidden_sizes, action_shape=action_shape, concat=True)
c2_model = Net(state_shape=obs_shape[0], hidden_sizes=hidden_sizes, action_shape=action_shape, concat=True)

# Create models for actor and both critics in addition to the optimizers for each model
#actor = DiscreteActor(preprocess_net=a_model, action_shape=action_shape, softmax_output=False)
actor = ContinuousActorProbabilistic(preprocess_net=a_model, action_shape=action_shape, unbounded=True)
actor_optim = AdamOptimizerFactory(lr=actor_lr)
#critic1 = DiscreteCritic(preprocess_net=c1_model, last_size=action_shape)
critic1 = ContinuousCritic(preprocess_net=c1_model)
critic1_optim = AdamOptimizerFactory(lr=critic_lr)
#critic2 = DiscreteCritic(preprocess_net=c2_model, last_size=action_shape)
critic2 = ContinuousCritic(preprocess_net=c2_model)
critic2_optim = AdamOptimizerFactory(lr=critic_lr)

# Define automatic alpha using the target entropy and the AutoAlpha class
#target_entropy = 0.5 * np.log(action_shape)
target_entropy = -1 * 2 #action dim
log_alpha = 0.0
alpha_optim = AdamOptimizerFactory(lr=alpha_lr)
alpha_param = AutoAlpha(target_entropy, log_alpha, alpha_optim)

# Define policy and algorithms
policy = SACPolicy(
    actor=actor,
    action_space=vector_env.action_space[0],
)

algorithm = SAC(
    policy=policy,
    policy_optim=actor_optim,
    critic=critic1,
    critic_optim=critic1_optim,
    critic2=critic2,
    critic2_optim=critic2_optim,
    tau=tau,
    gamma=gamma,
    alpha=alpha_param,
    n_step_return_horizon=n_step,
)

# Create the path to the log into
now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
log_path_base = r"C:\\Users\\Johan\\OneDrive\\Documents\School\\FAIRIS\\reinforcement_lib\\tianshou\Logs\\"
log_path = os.path.join(log_path_base, now)

# Create our tensorboard logger
writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer, training_interval=50, update_interval=1)

# Define Replay Buffer
buffer = ReplayBuffer(
    buffer_size,
)

#def hook(test, test2, step=[0]):
#    writer.add_scalar('action_0', test2['info']['action_0'], step=step[0])
#    writer.add_scalar('action_1', test2['info']['action_1'], step=step[0])
#    step[0] += 1

class Hooks:
    step = 0

    @staticmethod
    def on_step(action_batch, rollout_batch):
        writer.add_scalar('custom/action_0', rollout_batch['info']['action_0'], Hooks.step)
        writer.add_scalar('custom/action_1', rollout_batch['info']['action_1'], Hooks.step)
        writer.add_scalar('custom/imu', rollout_batch['info']['imu'], Hooks.step)
        writer.add_scalar('custom/reward', rollout_batch['info']['reward'], Hooks.step)
        Hooks.step += 1

#    print(f"test: {test}, test2: {test2}")

# Stop Function
def stop_fn(mean_reward):
    thres = -2.5
    print(f"Mean reward: {mean_reward}")
    if mean_reward > thres:
        return True
    else:
        return False

def standard_collector(env):
    return None

# Set Test
def test_collector(env):
    env.workers[0].env.set_test()


# Define collector
collector = Collector[CollectStats](
    standard_collector, algorithm, vector_env, buffer, exploration_noise=True, on_step_hook=Hooks.on_step,   
)
test_collector = Collector[CollectStats](
    test_collector, algorithm, vector_env, exploration_noise=True,
)

# Warmup our buffer with 4*batch_size
collector.reset()
collector.collect(n_step=4*batch_size)
# Train
result = algorithm.run_training(
    OffPolicyTrainerParams(
        training_collector=collector,
        test_collector=test_collector,
        stop_fn=stop_fn,
        max_epochs=epoch,
        epoch_num_steps=epoch_num_steps,
        collection_step_num_env_steps=collection_step_num_env_steps,
        batch_size=batch_size,
        update_step_num_gradient_steps_per_sample=update_per_step,
        logger=logger,
    ),
)
print(result)

if save_weights:
    torch.save(algorithm.state_dict(), weight_path)
