import numpy as np
import matplotlib.pyplot as plotter
import seaborn as sns
import gymnasium as gym
from gymnasium.envs.registration import register
import time as t
import os

def make_env(id_str = "FrozenLakeNotSlippery-v0"):
    try:
        register(
    id = id_str,
    entry_point = "gymnasium.envs.toy_text:FrozenLakeEnv",
    kwargs = {"map_name": "4x4", "is_slippery": False},
    max_episode_steps = 150,
    reward_threshold = 0.78
)
    except:
        print("Already registered gym environment")


make_env()
env = gym.make(id = "FrozenLakeNotSlippery-v0")
action_size = env.action_space.n
state_size = env.observation_space.n
EPISODES = int(2.5e04)
ALPHA = 0.8
GAMMA = 0.95
exploration = 1 #epsilon, i.e. to explore, or to choose the way known to the agent
max_exploration = 1.5
min_exploration = 0.01
decay_rate = 1e-3
initial_state = env.reset()
def makeqtable():
    #rows --> states, columns --> actions
    table = np.zeros([state_size, action_size])
    return table
qtable = makeqtable()

def greedy_selection(exploration, qtable, discrete_state, env = env):
    # choose whether to max future reward, or explore the game
    rand_num = np.random.random()
    if rand_num > exploration: #choose to maximize reward
        state_row = qtable[discrete_state[0], :]
        action = np.argmax(state_row)
    else:#choose a random action
        action = env.action_space.sample()
    return action

def compute_q_val(q_val, reward, optimal_qval):
    return q_val + ALPHA * (reward + GAMMA * optimal_qval - q_val)

def init_explore(speed = 0, limit = 20):
    for i in range(limit):
        action = env.action_space.sample()
        unpacked = env.step(action = action)
        t.sleep(speed)
        if unpacked[2] or unpacked[3]:
            break
    env.close()

# init_explore(limit = 100)

def reduce_exploration(eps):
    return min_exploration + (max_exploration - min_exploration) * np.exp(-decay_rate * eps)

rewards = []
logger = 1000
for eps in range(int(EPISODES)):
    state = env.reset()     
    total_rewards = 0
    done = False
        
    while not done:
        #do something
        action = greedy_selection(exploration, qtable, state)
        #get result of action
        unpacked = env.step(action = action)#                                      &
        #get current q_value                                                       |
        curr_qvalue = qtable[state[0], action]#                                    |
        new_state = [unpacked[0]]#                                                 |
        reward = unpacked[1]#                                                      |
        #get max qvalue for the state s we are at, after commiting an action here  |
        curr_opt_qval = np.max(qtable[new_state, :])
        #compute the q value at s+1
        next_optimal_q = compute_q_val(q_val = curr_qvalue, reward = reward, optimal_qval = curr_opt_qval)
        #update the table, by adding best q_val, rest remains as 0s
        qtable[state[0], action] = next_optimal_q
        #track rewards
        total_rewards = reward + total_rewards
        #update state
        state = new_state
        done = unpacked[2] or unpacked[3]
    #agent finishes a round
    eps += 1
    exploration = reduce_exploration(eps = eps)
    rewards.append(total_rewards)
    if eps % logger == 0:
        print(np.sum(rewards))
env.close()



#playing the game with the updated qtable

make_env(id_str = "FrozenLakeNotSlippery-v1")
env = gym.make(id = "FrozenLakeNotSlippery-v1", render_mode = "human")
state = env.reset()
for steps in range(100):
    action = np.argmax(qtable[state[0]])
    unpacker = env.step(action = action)
    if unpacker[2] or unpacker[3]:
        break
    t.sleep(0.2)
env.close()






# print(exploration)
# print(qtable)