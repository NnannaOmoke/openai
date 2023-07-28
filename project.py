import numpy as np
import gymnasium as gym
import time as t
# from continousspaces import discretize_observations, greedy_selection, reduce_exploration

environment = gym.make("MountainCar-v0")
BINS_SIZE = 50
ALPHA = 0.6
GAMMA = 0.97
EPS = int(2e04)
max_explr = 1.5
min_explr = 1e-2
decay_rate = 1e-3
explr = 1

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

def reduce_exploration(eps, min_explr = min_explr, max_explr = max_explr, decay_rate = decay_rate):
    return min_explr + (max_explr - min_explr) * np.exp(-decay_rate * eps)

def greedy_selection(exploration, qtable, discrete_state, env = environment):
    # choose whether to max future reward, or explore the game
    rand_num = np.random.random()
    if rand_num > exploration: #choose to maximize reward
        state_row = qtable[discrete_state[0]]
        action = np.argmax(state_row)
    else:#choose a random action
        action = env.action_space.sample()
    return action

def discretize_observations(observations, bins):
    binned = []
    for i, obs in enumerate(observations):
        discretized = np.digitize(x = obs, bins = bins[i])
        binned.append(discretized)
    return tuple(binned)



#main loop
#hopefully this works
logger = 500
cum_reward = 0
for _ in range(EPS):
   initial_state = environment.reset()
   discretized_init = discretize_observations(initial_state[0], BINS)
   done = False
   while not done:
        action = greedy_selection(exploration = explr, qtable = qtable, discrete_state = discretized_init, env = environment)
        results = environment.step(action = action)
        curr = discretize_observations(results[0], BINS)
        prev_qvalue = qtable[curr[0] + [action, ]]
        optimal_qvalue = np.max(qtable[curr])
        next_qvalue = compute_q_val(prev_qvalue, results[1], optimal_qvalue)
        qtable[curr[0] + [action, ]] = next_qvalue
        discretized_init = curr
        reward = results[1]
        done = results[2] or results[3]
        cum_reward += results[1]
   explr = reduce_exploration(_, min_explr, max_explr, decay_rate)
   if _ % logger == 0:
       print(reward)
environment.close()
