import numpy as np
from time import time
import rlunity
import gym

max_steps = 1000
n = 10

env = gym.make("UnityCarPixels-v0")

print(env)

ob = env.reset()

print('observation', np.shape(ob))

t_init = time()
global_step = 0

resets = 0

while True:
    ob, reward, done, env_info = env.step(env.action_space.sample())

    resets += np.sum(done)

    if done:
      ob = env.reset()

    global_step += 1

    if global_step > max_steps:
        break

fps = global_step / (time() - t_init)
print('steps per second {:.4f}'.format(fps))
