import numpy as np
import gymnasium as gym
import time as t
from continousspaces import discretize_observations, greedy_selection

environment = gym.make("MountainCar-v0", render_mode = "rgb_array_list")
BINS_SIZE = 50
ALPHA = 0.6
GAMMA = 0.97
EPS = int(2e04)

def make_bins():
    x_pos = np.linspace(-1.2, 0.6, BINS_SIZE)
    x_velocity = np.linspace(-0.07, 0.07, BINS_SIZE)
    bins = np.array([x_pos, x_velocity])
    return bins
BINS = make_bins()
Q_SHAPE = (BINS_SIZE, BINS_SIZE, environment.action_space.n)
qtable = np.zeros(Q_SHAPE)

def compute_q_val(q_val, reward, optimal_qval):
    return q_val + ALPHA * (reward + GAMMA * optimal_qval - q_val)


