import time as t
import matplotlib.pyplot as plotter
import numpy as np
import gymnasium as gym


env = gym.make("CartPole-v1")
start_vars = env.reset()
BINS_NUM = 10
EPOCHS = int(2e04)
ALPHA = 0.8
GAMMA = 0.95
exploration = 1
min_explr = 0.01
max_explr = 1.5
decay_rate = 1e-3
qtable_shape = (BINS_NUM, BINS_NUM, BINS_NUM, BINS_NUM, env.action_space.n)
qtable = np.zeros(qtable_shape)


def explore_env(speed = 0.01):
    obs = []
    for step in range(100):
        action = env.action_space.sample()
        unpacker = env.step(action = action)
        obs.append(unpacker[0])
        t.sleep(speed)
    env.close()
    return obs


def discretize_observations(observations, bins):
    binned = []
    for i, obs in enumerate(observations):
        discretized = np.digitize(x = obs, bins = bins[i])
        binned.append(discretized)
    return tuple(binned)


def create_bins(bins_per_observation = 10):
    x_position = np.linspace(-4.8, 4.8, bins_per_observation)
    cart_velocity = np.linspace(-5, -5, bins_per_observation)
    angles = np.linspace(-0.418, 0.418, bins_per_observation)
    angular_velocity = np.linspace(-5, -5, bins_per_observation)
    bins = np.array(
        [
           x_position,
           cart_velocity,
           angles,
           angular_velocity   
        ]
    )
    return bins

BINS = create_bins(BINS_NUM)

def greedy_selection(exploration, qtable, discrete_state, env = env):
    # choose whether to max future reward, or explore the game
    rand_num = np.random.random()
    if rand_num > exploration: #choose to maximize reward
        state_row = qtable[discrete_state[0]]
        action = np.argmax(state_row)
    else:#choose a random action
        action = env.action_space.sample()
    return action

def reduce_exploration(eps):
    return min_explr + (max_explr - min_explr) * np.exp(-decay_rate * eps)


def compute_q_val(q_val, reward, optimal_qval):
    return q_val + ALPHA * (reward + GAMMA * optimal_qval - q_val)


def failure(done, points, reward):
    if done and points < 150:
        reward -= 200
    return reward

#main

# print(start_vars)


logger = 500
# render_interval = int(10e03)
# fig = plotter.figure()
# ax = fig.add_subplot(111)
# plotter.ion()
# fig.canvas.draw()
# plotter.show()
reward_log = []
points_log = []
mean_points_log = []
epochs = []

for eps in range(EPOCHS):
    initial_state = env.reset()
    discretized_state = discretize_observations(initial_state[0], BINS)
    done = False
    points = 0
    
    epochs.append(eps)

    while not done:
        action = greedy_selection(exploration = exploration, qtable = qtable, env = env, discrete_state = discretized_state)
        unpacker = env.step(action = action)
        reward = failure(done = unpacker[2] or unpacker[3], points = points, reward = unpacker[1])
        current_discretized = discretize_observations([unpacker[0]], BINS)
        old_qvalue = qtable[discretized_state[0]] + (action, )
        optimal_qvalue = np.max(qtable[current_discretized])
        next_qvalue = compute_q_val(old_qvalue, reward, optimal_qvalue)
        qtable[discretized_state[0] + (action, )] = next_qvalue
        discretized_state_two = current_discretized
        points += 1
        reward_log.append(reward)
        done = unpacker[2] or unpacker[3]
    exploration = reduce_exploration(eps)
    points_log.append(points)
    running_mean = round(np.mean(points_log[-30: ]), 2)   
    mean_points_log.append(running_mean)
    if eps % logger == 0:
        print(np.sum(reward_log))
        # ax.clear()
        # ax.scatter(epochs, points_log)
        # ax.plot(epochs, points_log)
        # ax.plot(epochs, mean_points_log, label = f"Running Mean: {running_mean}")
        # plotter.legend()
        # fig.canvas.draw()
        # plotter.ion()
        # plotter.show()
env.close()

# print(explore_env(0.2))


new_env = gym.make("CartPole-v1", render_mode = "human")
observation = new_env.reset()
rewards = 0
for _ in range(1000):
    discrete_state = discretize_observations(observation[0], BINS)  # get bins
    action = np.argmax(qtable[discrete_state])  # and chose action from the Q-Table
    unpackerr = new_env.step(action) # Finally perform the action
    rewards += 1
    if unpackerr[2] or unpackerr[3]:
        print(f"You got {rewards} points!")
        break
new_env.close()