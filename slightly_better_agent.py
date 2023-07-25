import gymnasium as gym
import time as t
import matplotlib.pyplot as plotter
import datetime as dt
# def log_seed():
#     open("seed.txt", mode = "w").write("101")
# def read_seed():
#     seed = open("seed.txt", mode = "r").read()
#     return int(seed)
# seed = read_seed()
env = gym.make("MountainCar-v0", render_mode = "human")
no_avalaible_actions = env.action_space.n
reset_vals = env.reset()

# env.seed(seed)

def explore(limit = 20, action = 1, verbose = False, log_states = False, speed = 0.1):
    env.reset()
    if(log_states):
        open("explorationlog.txt", mode = "a").write(dt.datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + "\n")
    for i in range(limit):
        unpacker = env.step(action = action)
        if(verbose):
            if(log_states):
                open("explorationlog.txt", mode = "a").write(
                f"\tObservation: {unpacker[0]}\n\tReward: {unpacker[1]}\n\tDone/Terminated: {unpacker[2]}\n"
                )
            else:
                print(
                f"Observation: {unpacker[0]}\nReward: {unpacker[1]}\nDone/Terminated: {unpacker[2]}"
            )
        t.sleep(speed)
    env.close()
    return 
#grabs observation and does something off that
def simple_agent(observations):
    position = observations[0][0]
    velocity = observations[0][1]
    if -0.1 > position > 0.4:
        action = 2
    elif velocity < 0 and position < 0.2:
        action = 0
    else:
        action = 1
    return action

def agent_based_explorer(limit = 20, verbose = False, log_states = False, speed = 0.1):
    init = env.reset()
    if(log_states):
        open("explorationlog.txt", mode = "a").write(dt.datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + "\n")
    for i in range(limit):
        unpacker = env.step(action = simple_agent(init))
        init = [unpacker[0]]
        if(verbose):
            if(log_states):
                open("explorationlog.txt", mode = "a").write(
                f"\tObservation: {unpacker[0]}\n\tReward: {unpacker[1]}\n\tDone/Terminated: {unpacker[2]}\n"
                )
            else:
                print(
                f"Observation: {unpacker[0]}\nReward: {unpacker[1]}\nDone/Terminated: {unpacker[2]}"
            )
        t.sleep(speed)
    env.close()
    return 

agent_based_explorer(limit = 600, speed = 0, verbose = True, log_states = True)