# Import Required Libs
import torch
import numpy as np
import gymnasium as gym
import math

# FAIRIS libs
from fairis_lib.robot_lib import hambot

class FAIRISEnv(gym.Env):
    def __init__(self, maze_files, horizont, device="cpu"):

        # Env Variables
        self.robot = hambot.HamBot(use_camera=False)
        self.maze_files = maze_files
        self.current_maze_index = 0
        self.maze_file = self.maze_files[self.current_maze_index]
        self.first_run = True # Var to see if we need to load maze
        self.horizon = horizont
        self.length = 0
        self.max_lidar = 20
        self.lidar_noise = 0
        self.xy_noise = 0
        self.linear_noise = 0
        self.action_noise = 0
         # Number of discrete actions and their corresponding headings (degrees)
        self.N_ACTIONS = 8
        self.ACTION_ANGLES_DEG = np.array([i * (360 / self.N_ACTIONS) for i in range(self.N_ACTIONS)])  # [0, 45, 90, ..., 315]
 
#        self.observation_space_size = 2
#        self.action_space_size = 8

        self.observation_space = gym.spaces.Box(low=-20, high=20, shape=(364,))
        self.action_space = gym.spaces.Discrete(8)
        
    def switch_maze(self):
        self.current_maze_idx = (self.current_maze_index + 1) % len(self.maze_files)
        self.maze_file = self.maze_files[self.current_maze_index]
        self.robot.load_environment(self.maze_file)
                
    def add_xynoise(self, x, y, mean=0, std_dev=0.1, clip_amount=5):
        noise_x = np.random.normal(mean, std_dev)
        noise_y = np.random.normal(mean, std_dev)
        x_noisy = np.clip(x + noise_x, -1*clip_amount, clip_amount)
        y_noisy = np.clip(y + noise_y, -1*clip_amount, clip_amount)
        return x_noisy, y_noisy
    
    def add_lidar_noise(self, lidar_readings):
        noisy_readings = []
        for reading in lidar_readings:
            sigma = np.random.uniform(0.005, 0.05)
            noise = np.random.normal(0, sigma)
            noisy_readings.append(max(0, reading + noise))
        return np.array(noisy_readings)
    
    def add_linear_acceleration_noise(self,linear_accel, sigma_range=(0.01, 0.1), clip_range=(-5.0, 5.0)):
        sigma = np.random.uniform(*sigma_range)
        noise = np.random.normal(0, sigma, size=np.shape(linear_accel))
        return np.clip(np.array(linear_accel) + noise, *clip_range)
 
    def add_action_angle_noise(self, action, sigma_range=(2.5, 15.0)):
        sigma_deg = np.random.uniform(*sigma_range)
        intended_angle_deg = self.ACTION_ANGLES_DEG[action]
        perturbed_angle_deg = intended_angle_deg + np.random.normal(0, sigma_deg)
    
        # Wrap to [0, 360) and find closest discrete action
        perturbed_angle_deg = perturbed_angle_deg % 360
        deltas = np.abs(self.ACTION_ANGLES_DEG - perturbed_angle_deg)
        # Handle wrap-around (e.g. distance between 350 deg and 10 deg)
        deltas = np.minimum(deltas, 360 - deltas)
        return int(np.argmin(deltas))

    def reset(self, seed=None, options=None):
        
        self.switch_maze()
        
        if self.first_run:
            self.robot.load_environment(self.maze_file)
            self.first_run = False

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

        # Do action
        if(self.action_noise):
            action = self.add_action_angle_noise(action)
        
        value = self.robot.perform_action_with_PID(int(action))

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

        return state, reward, done, False, info

    def getState(self):
        robot_x, robot_y, robot_theta = self.robot.get_robot_pose()
        if self.xy_noise:
            robot_x, robot_y = self.add_xynoise(robot_x, robot_y)

        goal_x, goal_y = self.robot.maze.get_goal_location()

        lidar_image = self.robot.get_lidar_range_image()
        if self.lidar_noise:
            lidar_image = self.add_lidar_noise(lidar_image)
        
        for idx in range(len(lidar_image)):
            if lidar_image[idx] > 20:
                lidar_image[idx] = 20

#        print(lidar_image.shape)

        return np.concatenate(([robot_x, robot_y], lidar_image, [goal_x, goal_y]))
