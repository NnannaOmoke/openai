import gymnasium as gym
import matplotlib.pyplot as plotter
import time as t
from gym.utils import play
r_modes = ["human", "rgb_array"]
env = gym.make("Breakout-v4", render_mode = r_modes[0])
# print(env.action_space.n)

def step_one():
    env.reset()
    for step in range(200):
        env.step(action = 0) 
def try_actions(limit = 200):
    env.reset()
    for i in range(limit):
        random_action = env.action_space.sample()
    return random_action
def implement_actions(time_limit = 200, verbose = 1):
    env.reset()
    for i in range(time_limit):
        random_action = env.action_space.sample()
        test = env.step(action = random_action)
        if(verbose == 0):
            print(
            f"Reward: {test[1]}\nDone: {test[2]}\nInfo: {test[4]}"
        )
        t.sleep(0.1)
    env.close()
    return
implement_actions(200)
# plotter.imshow(env.render())
# plotter.show()
#testing 
# for step in range(200):
#     env.render()
#     random_action = env.action_space.sample()
# print(random_action)
