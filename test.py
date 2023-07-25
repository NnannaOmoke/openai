import gymnasium as gym
import matplotlib.pyplot as plotter
import time as t
from gym.utils import play
env = gym.make("Breakout-v4", render_mode = "human")
keep_reset = env.reset()
# print(env.action_space.n)

def step_one():
    for step in range(200):
        env.step(action = 0) 
def try_actions(limit = 200):
    for i in range(limit):
        random_action = env.action_space.sample()
        return random_action
print(try_actions())
# plotter.imshow(env.render())
# plotter.show()
#testing 
# for step in range(200):
#     env.render()
#     random_action = env.action_space.sample()
# print(random_action)
