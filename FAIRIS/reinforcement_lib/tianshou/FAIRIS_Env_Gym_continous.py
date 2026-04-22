# Import Required Libs
import torch
import numpy as np
import gymnasium as gym
import math

# FAIRIS libs
from fairis_lib.robot_lib import hambot

class FAIRISEnv(gym.Env):
    def __init__(self, maze_filet, horizont, device="cpu", continuous=False, continuous_vel_dur=8, min_length=0.2, test_maze=None):


        # #Domain Randomization variables - added in for testing DR
        self.lidar_sigma = 0.0
        self.pos_sigma = 5.0
        self.motor_sigma = 0.0

        # Env Variables
        self.robot = hambot.HamBot(use_camera=False)
        self.maze_file = maze_filet
        self.first_run = True # Var to see if we need to load maze
        self.horizon = horizont
        self.length = 0
        self.max_lidar = 20
        self.continuous = continuous
        self.max_motor = 10
        self.vel_dur = continuous_vel_dur
        self.min_len = min_length
        self.past_thres = 0.0001
#        self.observation_space_size = 2
#        self.action_space_size = 8
        self.test_next = False
        self.test_maze = test_maze
        if self.test_maze == None:
            self.test_maze = self.maze_file

        self.observation_space = gym.spaces.Box(low=-20, high=20, shape=(365,))
        if continuous:
            self.action_space = gym.spaces.Box(low=-1*self.max_motor, high=self.max_motor, shape=(2,))
        else:
            self.action_space = gym.spaces.Discrete(8)

    def set_test(self):
        self.test_next = True
#        print("Called")
    # #Added DR functions to create noise
    def lidar_noise(self, lidar_readings):
        noise = np.random.normal(0, self.lidar_sigma, size=len(lidar_readings))
        return np.maximum(0, lidar_readings + noise)
    
    def pos_noise(self, x, y):
        x += np.random.normal(0, self.pos_sigma)
        y += np.random.nomral(0, self.pos_sigma)
        return x, y

    def motor_noise(self, action):
        noise = np.random.normal(0, self.motor_sigma, size=2)
        return np.clip(action + noise, -self.max_motor, self.max_motor) 
     ######   

    def reset(self, seed=None, options=None):
        if self.first_run:
            self.robot.load_environment(self.maze_file)
            self.first_run = False

        if self.test_next:
            self.robot.load_environment(self.test_maze)
            self.first_run = True
            self.test_next = False

        self.robot.move_to_random_experiment_start()
        self.robot.experiment_supervisor.simulationResetPhysics()

        # Get first state
        state = self.getState()
        self.past_xy = state[0:2]

        self.length = 0

        info = {}
#        print("Reset")

        return state, info

    def step(self, action):
        done = False
        reward = -0.5
        #print(f"action: {action}")

        # Do action
        if self.continuous:
            if self.motor_sigma > 0: # Puts noise on the motors during step state
                action = self.motor_noise(action)

            self.robot.left_motor.setVelocity(action[0])
            self.robot.right_motor.setVelocity(action[1])
#            self.robot.left_motor.setVelocity(-5)
#            self.robot.right_motor.setVelocity(5)
            cur_dur = 0
#            while (self.robot.experiment_supervisor.step(self.robot.timestep) != -1) and (cur_dur < self.vel_dur):
#                cur_dur += self.robot.timestep
            if self.robot.experiment_supervisor.step(self.robot.timestep * self.vel_dur) == -1:
                print("Error")
            value = 0
        else:
            value = self.robot.perform_action_with_PID(int(action))

        # Get new state
#        print(f"actions: {action}")
        state = self.getState()

        # Calculate reward
        if not(self.continuous):
            if self.robot.check_at_goal():
                reward = 10.0
                done = True
            elif self.length >= self.horizon:
                done = True
                reward = -1.0
            elif value == -1:
                reward = -1.0
            else:
#                reward = 0
                goal_x, goal_y = self.robot.maze.get_goal_location()
                dist = math.sqrt((goal_x - state[0]) ** 2 + (goal_y - state[1]) ** 2) / 7
                reward = -0.1 - dist
                #reward = -0.5
        else:
            if self.robot.check_at_goal():
                reward = 10.0
                done = True
            elif self.length >= self.horizon:
                done = True
                reward = -1.0
            elif (min(state[5:]) < self.min_len):
                done = True
                reward = -20.0
            else:
                done = False
                goal_x, goal_y = self.robot.maze.get_goal_location()
                dist = math.sqrt((goal_x - state[0]) ** 2 + (goal_y - state[1]) ** 2) / 7
                last_pen = 0
                if ((self.past_xy[0] - state[0]) ** 2 + (self.past_xy[1] - state[1]) ** 2) < self.past_thres:
                    last_pen = -0.2
#                    print(f"pen: {action[0]}, {action[1]}")
                reward = -0.1 - dist + last_pen

        self.length += 1

        if self.continuous:
            info = {"action_0": action[0], "action_1": action[1], "imu": state[4], "reward": reward}
        else:
            info = {}

#        print(f"done: {done}")

        return state, reward, done, False, info

    def getState(self):
        robot_x, robot_y, robot_theta = self.robot.get_robot_pose()

        goal_x, goal_y = self.robot.maze.get_goal_location()

        imu_reading = self.robot.imu.getRollPitchYaw()[-1]
#        print(f"imu: {imu_reading}")
#        acc_reading = self.robot.accelerometer.getValues()
#        print(f"acc: {acc_reading}")

        lidar_image = self.robot.get_lidar_range_image()
        for idx in range(len(lidar_image)):
            if lidar_image[idx] > 20:
                lidar_image[idx] = 20

#        print(lidar_image.shape)

        return np.array([robot_x, robot_y] + [goal_x, goal_y] + [imu_reading] + lidar_image)
