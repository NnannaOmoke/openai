import numpy as np
import matplotlib.pyplot as plotter
import seaborn as sns
import gymnasium as gym
from gymnasium.envs.registration import register
import time as t

def make_env():
    try:
        register(
    id = "FrozenLakeNotSlippery-v0",
    entry_point = "gymnasium.envs.toy_text:FrozenLakeEnv",
    kwargs = {"map_name": "4x4", "is_slippery": False},
    max_episode_steps = 150,
    reward_threshold = 0.78
)
    except:
        print("Already registered gym environment")

make_env()
env = gym.make(id = "FrozenLakeNotSlippery-v0", render_mode = "human")
action_size = env.action_space.n
state_size = env.observation_space.n
initial_state = env.reset()

def init_explore(speed = 0, limit = 20):
    for i in range(limit):
        action = env.action_space.sample()
        unpacked = env.step(action = action)
        t.sleep(speed)
        if unpacked[2] or unpacked[3]:
            break
    env.close()

init_explore(limit = 100)

def makeqtable():
    #rows --> states, columns --> actions
    table = np.zeros([state_size, action_size])
