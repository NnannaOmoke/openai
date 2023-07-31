import numpy as np
import random as r
import gymnasium as gym
from collections import deque
from keras.models import Sequential, clone_model
from keras.layers import Dense, Activation
from keras.optimizers import Adam

env = gym.make("CartPole-v1", render_mode = "human")
initials = env.reset()

def explore_env():
    for _ in range(200):
        env.step(env.action_space.sample())
    return