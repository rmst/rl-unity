"""
THIS IS CURRENTLY NOT WORKING!
"""



import gym
from gym import spaces

from rlunity.unity_env import UnityEnv, logger

import numpy as np


class UnityCar(UnityEnv):
  """
  a = [steering wheel (right positive), throttle]
  """
  def __init__(self):
    super().__init__(w=128, h=128, batchmode=False)
    self.t_max = 100000  # about 1h of driving at 30fps

    self.t0 = 7*60

    sbm = self.sbm + 33
    self.observation_space = spaces.Box(-np.ones([sbm]), np.ones([sbm]))

  def process_raw_state(self, raw_state):
    logger.debug("Distance = " + str(raw_state[0]) + " ; Speed along road = " + str(raw_state[1]))
    logger.debug("Position = " + str(raw_state[2:5]) + " ; Projection = " + str(raw_state[5:8]))
    logger.debug("Collision detected : " + ("True" if raw_state[8] == 1.0 else "False"))
    logger.debug("Road direction : " + str(raw_state[9:12]) + "; Car direction : " + str(raw_state[12:15]))
    logger.debug("Next angle :" + str(raw_state[15]))

    # radial basis function features
    pos = self.wp - raw_state[2:5]
    pos_fp = np.exp(- np.sum(np.square(pos), axis=1) / 100)

    direction = raw_state[12:15] - raw_state[9:12]
    self.v[self.t] = raw_state[1]
    av_speed = self.v[max(0, self.t - self.t0):self.t+1].mean()

    return np.concatenate(((
        raw_state[0] / 100,  # distance from road center (in meters) TODO: verify
        raw_state[1],  # speed projected onto road (in meters / s) TODO: verify
        av_speed,  # running average of the speed
      ),
      direction,  # direction relative to road
      pos_fp,  # radial basis functions of location w.r.t. the waypoints
    ))

  def _reset(self):
    self.v = np.zeros(self.t_max)
    self.t = 0
    state, frame = super()._reset()
    state = self.process_raw_state(state)
    return state

  def _step(self, action):
    action = np.clip(action, -1, 1)
    self.send(action)
    state, frame = self.receive()

    state = self.process_raw_state(state)

    distance = state[0]
    speed = state[1]
    av_speed = state[2]

    direction = state[3:6]

    # logger.info(f'd0 = {d0}')
    done = np.abs(distance) > 1.5 or (self.t > self.t0 and av_speed < 0.1)

    # r_speed = speed
    r_speed = 0.1 - .9 * (speed - .2)**2

    reward = .2 - 1 * (distance - .5) ** 2 + r_speed
    if done:
      reward -= 30

    reward = 1 * reward

    done = done or self.t+1 >= self.t_max

    self.t += 1

    return state, reward, done, {}

  def report(self):
    logger.info(f'Distance driven: {self.v.sum()}')


def test_unity_car():
  import logging
  import rlunity
  logger.setLevel(logging.DEBUG)

  env = gym.make('UnityCar-v0')  # requires import rlunity
  env.unwrapped.conf(loglevel='debug', log_unity=True)
  env.reset()
  for i in range(10000):
    print(i)
    env.step([.0, 1.0])

    if (i + 1) % 300 == 0:
      env.reset()


if __name__ == '__main__':
  test_unity_car()
