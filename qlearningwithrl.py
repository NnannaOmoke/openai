import gymnasium as gym
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import Agent

from introdqn import explore_env

env = gym.make("CartPole-v1", render_mode = "human")
explore_env(env)