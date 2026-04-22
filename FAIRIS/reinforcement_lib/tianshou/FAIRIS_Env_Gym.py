# # Import Required Libs
# import torch
# import numpy as np
# import gymnasium as gym
# import math

# # FAIRIS libs
# from fairis_lib.robot_lib import hambot

# class FAIRISEnv(gym.Env):
#     def __init__(self, maze_filet, horizont, device="cpu"):

#         # Env Variables
#         self.robot = hambot.HamBot(use_camera=False)
#         self.maze_file = maze_filet
#         self.first_run = True # Var to see if we need to load maze
#         self.horizon = horizont
#         self.length = 0
#         self.max_lidar = 20
# #        self.observation_space_size = 2
# #        self.action_space_size = 8

#         self.observation_space = gym.spaces.Box(low=-20, high=20, shape=(364,))
#         self.action_space = gym.spaces.Discrete(8)

#     def reset(self, seed=None, options=None):
#         if self.first_run:
#             self.robot.load_environment(self.maze_file)
#             self.first_run = False

#         self.robot.move_to_random_experiment_start()
#         self.robot.experiment_supervisor.simulationResetPhysics()

#         # Get first state
#         state = self.getState()

#         self.length = 0

#         info = {}

#         return state, info

#     def step(self, action):
#         done = False
#         reward = -0.5

#         # Do action
#         value = self.robot.perform_action_with_PID(int(action))

#         # Get new state
#         state = self.getState()

#         # Calculate reward
#         if self.robot.check_at_goal():
#             reward = 10.0
#             done = True
#         elif self.length >= self.horizon:
#             done = True
#             reward = -1.0
#         elif value == -1:
#             reward = -1.0
#         else:
# #            reward = 0
#             goal_x, goal_y = self.robot.maze.get_goal_location()
#             dist = math.sqrt((goal_x - state[0]) ** 2 + (goal_y - state[1]) ** 2) / 7
#             reward = -0.1 - dist
#             #reward = -0.5

#         self.length += 1

#         info = {}

#         return state, reward, done, False, info

#     def getState(self):
#         robot_x, robot_y, robot_theta = self.robot.get_robot_pose()

#         goal_x, goal_y = self.robot.maze.get_goal_location()

#         lidar_image = self.robot.get_lidar_range_image()
#         for idx in range(len(lidar_image)):
#             if lidar_image[idx] > 20:
#                 lidar_image[idx] = 20

# #        print(lidar_image.shape)

#         return np.array([robot_x, robot_y] + lidar_image + [goal_x, goal_y])



#######################################################################################

# Import Required Libs
import torch
import numpy as np
import gymnasium as gym
import math
import os
import sys
import random
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'BrendonSAC'))
from domain_ran import (
    linear_acc_noise,
    action_noise,
    lidar_noise,
    pos_noise
)

# FAIRIS libs
from fairis_lib.robot_lib import hambot

class FAIRISEnv(gym.Env):
    def __init__(self, maze_file, horizont, device="cpu", dr_config = None):

        self.episode_counter = 0 #for maze switching
        self.maze_index = 1 
        self.episode_done = True #real episode check

        # Env Variables
        self.robot = hambot.HamBot(use_camera=False)
        self.maze_file = maze_file
        self.first_run = True # Var to see if we need to load maze
        self.horizon = horizont
        self.length = 0
        self.max_lidar = 20
#        self.observation_space_size = 2
#        self.action_space_size = 8

        self.dr = dr_config
        self.domain_randomization = dr_config.enabled if dr_config else False

        self.observation_space = gym.spaces.Box(low=-20, high=20, shape=(364,))
        self.action_space = gym.spaces.Discrete(8)

    def reset(self, seed=None, options=None):
        if self.first_run:
            self.robot.load_environment(self.maze_file)
            self.first_run = False


        #load the maze switch given every 10 episodes
        elif (self.domain_randomization 
              and self.dr.maze_freq_sw > 0 
              and len(self.dr.maze_files) > 1 #guard if maze list isn't set
              and self.episode_counter % self.dr.maze_freq_sw == 0):
            new_maze = self.dr.maze_files[self.maze_index % len(self.dr.maze_files)] #can also use 'random.choice' instead of sequential switch
            self.maze_index += 1
            self.robot.reset_environment() #clear old maze
            self.robot.load_environment(new_maze) #switch new one

        if self.episode_done: #added to make sure no episode was being skipped due to possible mid reset
            self.episode_counter += 1
        self.episode_done = False

        self.robot.move_to_random_experiment_start()
        self.robot.experiment_supervisor.simulationResetPhysics()

        # Get first state
        state = self.getState()

        self.length = 0

        info = {}

        return state, info

    def step(self, action):
        done = False
        reward = -0.5

        # Action angle noise — may redirect to adjacent discrete action
        if self.domain_randomization:
            action = action_noise(int(action), sigma_range=self.dr.action_sigma_range)
            # Linear acceleration noise — perturb how far the robot moves
            original_length = self.robot.action_set[action][1]
            noisy_length = float(linear_acc_noise(
                original_length, sigma_range=self.dr.lin_acc_sigma_range, clip_range=self.dr.lin_acc_clip_range
            ))
            self.robot.action_set[action][1] = noisy_length

        # Do action
        value = self.robot.perform_action_with_PID(int(action))

        if self.domain_randomization:
            self.robot.action_set[action][1] = original_length

        # Get new state
        state = self.getState()

        # Calculate reward
        if self.robot.check_at_goal():
            reward = 10.0
            done = True
        elif self.length >= self.horizon:
            done = True
            reward = -1.0
        elif value == -1:
            reward = -1.0
        else:
#            reward = 0
            goal_x, goal_y = self.robot.maze.get_goal_location()
            dist = math.sqrt((goal_x - state[0]) ** 2 + (goal_y - state[1]) ** 2) / 7
            reward = -0.1 - dist
            #reward = -0.5

        self.length += 1

        info = {}

        self.episode_done = done #return for real episode 

        return state, reward, done, False, info

    def getState(self):
        robot_x, robot_y, robot_theta = self.robot.get_robot_pose()

        goal_x, goal_y = self.robot.maze.get_goal_location()

        lidar_image = self.robot.get_lidar_range_image()
        for idx in range(len(lidar_image)):
            if lidar_image[idx] > 20:
                lidar_image[idx] = 20

        if self.domain_randomization:
            lidar_image = list(lidar_noise(lidar_image, sigma_range = self.dr.lidar_sigma_range))
        if self.domain_randomization:
            robot_x, robot_y = pos_noise(robot_x, robot_y, sigma_range = self.dr.pos_sigma_range)

        # if self.domain_randomization:
        #     lidar_image = list(lidar_noise(lidar_image))
        # if self.domain_randomization:
        #     robot_x, robot_y= pos_noise(robot_x, robot_y)

        
#        print(lidar_image.shape)

        return np.array([robot_x, robot_y] + lidar_image + [goal_x, goal_y])
