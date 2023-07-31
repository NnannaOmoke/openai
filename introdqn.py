import numpy as np
import random as r
import gymnasium as gym
from collections import deque
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import Adam

BATCH_SIZE = 32
EPISODES = 250
REDUCTION = 9.95e-1
L_RATE = 1e-3
GAMMA = 9.5e-1

replay_buffer = deque(maxlen = int(2e4))
explr = 1
update_target_learner = 10

env = gym.make("CartPole-v1")
initials = env.reset()
observation_nums = env.observation_space.shape[0]
num_actions = env.action_space.n


def explore_env():
    for _ in range(200):
        env.step(env.action_space.sample())
    return

def build_model():
    learner = Sequential()
    learner.add(Dense(input_shape = (1, observation_nums), activation = "relu", units = 16))
    learner.add(Dense(units = 32, activation = "relu"))
    learner.add(Dense(num_actions, activation = "linear"))
    # print(learner.summary())
    return learner

def build_target_network(learner):
    return clone_model(learner)
    
def greedy_selection(learner, explr, obs):
    if np.random.random() > explr:
        prediction = learner.predict(obs, verbose = 0)
        action = np.argmax(prediction)
    else:
       action = env.action_space.sample() 
    return action

def replay(replay_buffer, batch_size, learner, target_learner):
    if len(replay_buffer) < batch_size:
        return 
    samples = r.sample(replay_buffer, batch_size) #batch_size number of elements from a replay_buffer
    target_batch = []
    zipped_samples = list(zip(*samples))
    one, two, three, four, five = zipped_samples
    targets = target_learner.predict(np.array(one), verbose = 0)
    q_vals = learner.predict(np.array(four), verbose = 0)
    for index in range(batch_size):
        q_values = max(q_vals[index][0])
        target = targets[index].copy()
        if five[index]:
            target[0][two[index]] = three[index]
        else:
            target[0][two[index]] = three[index] + q_values * GAMMA
        target_batch.append(target)
    learner.fit(np.array(one), np.array(target_batch), epochs = 1, verbose = 0)

def update_model(epoch, update_target_learner, learner,  target_learner):
    if epoch > 0 and epoch % update_target_learner == 0:
        target_learner.set_weights(learner.get_weights())
    return   


def play_with_model(learner):
    new_env = gym.make("Cartpole-v1", render_mode = "human")
    init_status = new_env.reset()
    init_status = init_status[0].reshape((1, 4))
    for _ in range(350):
        action = np.argmax(learner.predict(init_status))
        unpacker = new_env.step(action = action)
        if unpacker[2] or unpacker[3]:
            break
    env.close()
    return

learner = build_model()
target_learner = build_target_network(learner)
learner.compile(loss = "mse", optimizer = Adam(learning_rate = L_RATE))
# print(learner.summary())

accumulated_reward = 0

for _ in range(EPISODES):
    init_state = env.reset()
    init_state = init_state[0].reshape([1, 4])
    done = False
    points = 0
    while not done:
        action = greedy_selection(learner, explr, init_state)
        unpacked = env.step(action = action)
        next_init= unpacked[0].reshape([1,4])
        replay_buffer.append((init_state, action, unpacked[1], next_init, unpacked[2] or unpacked[3]))
        init_state = next_init
        points += 1
        replay(replay_buffer, BATCH_SIZE, learner, target_learner)
        done = unpacked[2] or unpacked[3]
    explr *= REDUCTION
    update_model(_, update_target_learner, learner, target_learner)
    if points > accumulated_reward:
        accumulated_reward = points
    if _ % 25 == 0:
        print(f"{_}: POINTS: {points}, eps: {explr}, best so far: {accumulated_reward}")
env.close()


play_with_model(learner)