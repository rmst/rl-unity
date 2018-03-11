import gym
from PIL import Image
from gym import spaces

from rlunity.unity_env import UnityEnv, logger

import numpy as np


class UnityCarPixels(UnityEnv):
  """
  a = [steering wheel (right positive), throttle]
  """
  def __init__(self):
    super().__init__(batchmode=False)
    self.t_max = 20 * 60 * 10

    self.t0 = 20 * 7

    self.observation_space = spaces.Box(0, 255, shape=[84, 84])
    self.reward_range = (-.1, .1)
    self.r = 0

  def process_raw_state(self, raw_state):
    logger.debug("Distance = " + str(raw_state[0]) + " ; Speed along road = " + str(raw_state[1]))
    logger.debug("Position = " + str(raw_state[2:5]) + " ; Projection = " + str(raw_state[5:8]))
    logger.debug("Collision detected : " + ("True" if raw_state[8] == 1.0 else "False"))
    logger.debug("Road direction : " + str(raw_state[9:12]) + "; Car direction : " + str(raw_state[12:15]))

    direction = raw_state[12:15] - raw_state[9:12]
    self.v[self.t] = raw_state[1]
    av_speed = self.v[max(0, self.t - self.t0):self.t+1].mean()

    return np.concatenate(((
        raw_state[0] / 100,  # distance from road center (in meters) TODO: verify
        raw_state[1],  # speed projected onto road (in meters / s) TODO: verify
        av_speed,  # running average of the speed
      ),
      direction,  # direction relative to road
    ))

  def reset(self):
    self.v = np.zeros(self.t_max)
    self.t = 0
    state, frame = super().reset()
    state = self.process_raw_state(state)
    return self.proc_frame(frame)

  def proc_frame(self, frame):
    if np.shape(frame) != (84, 84, 3):
      img = Image.fromarray(frame)
      img = img.resize([84, 84]).convert('L')
      frame = np.asarray(img, dtype=np.uint8)

    return frame

  def step(self, action):
    action = np.clip(action, -1, 1)
    self.send(action)
    state, frame = self.receive()

    state = self.process_raw_state(state)

    distance = state[0]
    speed = state[1]
    av_speed = state[2]

    direction = state[3:6]

    # logger.info(f'd0 = {d0}')
    done = np.abs(distance) > 1.5 or (self.t > self.t0 and av_speed < 0.005)

    # r_speed = speed
    self.rs = rs = 1. * (speed - .4) ** 2
    # self.rd = rd = .01 * (distance - .5) ** 2
    self.rd = rd = .01 * (distance - .25) ** 2  # approx. right lane of the road
    reward = .05 - rd + 0.01 - rs
    # if done:
    #   reward -= 10


    self.r = np.clip(reward, *self.reward_range)
    done = done or self.t+1 >= self.t_max

    self.t += 1

    return self.proc_frame(frame), self.r, done, {'distance_from_road': distance, 'speed': speed, 'average_speed': av_speed, 'unwrapped_reward': reward}

  def report(self):
    logger.info(f'Distance driven: {self.v.sum()}')


def draw_rect(a, pos, color):
  """
  :param a:
  :param pos: (x0, y0), (x1, y1) in normalized cartesian coordinates
  :param color:
  :return:
  """
  pos = np.asarray(pos) * (1, -1) + (0, 1)  # transform to matrix coordinates (flip y axis)
  pos = pos * (a.shape[1], a.shape[0])
  (x0, y1), (x1, y0) = np.asarray(pos, dtype=np.int32)
  a[y0:y1, x0:x1, ...] = color
  return a

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='Unity Gym Environment')
  parser.add_argument('--batchmode', action='store_true', help='Run the simulator in batch mode with no graphics')
  args = parser.parse_args()
  logger.debug('Batchmode ' + str(args.batchmode))
  bm = args.batchmode

  import rlunity

  env = gym.make('UnityCarPixels-v0')  # requires import rlunity
  env.unwrapped.conf(loglevel='debug', log_unity=True)
  env.reset()
  for i in range(10000):
    print(i)
    env.step([.0, 1.0])

    if (i + 1) % 300 == 0:
      env.reset()

