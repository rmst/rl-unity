import threading
import numpy as np
import socket
import subprocess
import os
import gym
from time import sleep
import json
import sys
from gym import spaces
import logging

logger = logging.getLogger('UnityEnv')


class UnityEnv(gym.Env):
  """A base class for environments using Unity3D
  Implements the gym.Env interface. See
  https://gym.openai.com/docs
  and
  https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym
  """
  metadata = {'render.modes': ['human', 'rgb_array'],
              'video.frames_per_second': 20}

  def __init__(self, batchmode=False):
    self.proc = None
    self.soc = None
    self.connected = False

    self.ad = 2
    self.sd = 16
    self.batchmode = batchmode
    self.wp = None

    self.action_space = spaces.Box(-np.ones([self.ad]), np.ones([self.ad]))
    self.log_unity = False
    self.logfile = None
    self.restart = False
    self.configured = False

  def conf(self, loglevel='INFO', log_unity=False, logfile=None, w=128, h=128, *args, **kwargs):
    logger.setLevel(getattr(logging, loglevel.upper()))
    self.log_unity = log_unity
    if logfile:
      self.logfile = open(logfile, 'w')

    assert w >= 100 and h >= 100, 'the simulator does not support smaller resolutions than 100 at the moment'
    self.w = w
    self.h = h
    self.configured = True

  def connect(self):
    self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = '127.0.0.1'
    port = get_free_port(host)
    logger.debug('Port: {}'.format(port))
    assert port != 0
    import platform
    logger.debug('Platform ' + platform.platform())
    pl = 'windows' if 'Windows' in platform.platform() else 'unix'
    self.sim_path = os.path.join(os.path.dirname(__file__), '..', 'simulator', 'bin', pl)
    if (pl == 'windows'):
      bin = os.path.join(os.path.dirname(__file__), '..', 'simulator', 'bin', pl, 'sim.exe')
    else:
      bin = os.path.join(os.path.dirname(__file__), '..', 'simulator', 'bin', pl, 'sim.x86_64')
    bin = os.path.abspath(bin)
    env = os.environ.copy()

    env.update(
      RL_UNITY_PORT=str(port),
      RL_UNITY_WIDTH=str(self.w),
      RL_UNITY_HEIGHT=str(self.h),
      # MESA_GL_VERSION_OVERRIDE=str(3.3),
    )  # insert env variables here

    logger.debug('Simulator binary' + bin)

    # ensure that the sim doesn't read or write any cache or config files
    # TODO: only works on linux
    config_dir = os.path.expanduser('~/.config/unity3d/DefaultCompany/rl-unity')
    if os.path.isdir(config_dir):
      from shutil import rmtree
      rmtree(config_dir, ignore_errors=True)

    output_redirect = self.logfile if self.logfile else (subprocess.PIPE if self.log_unity else subprocess.DEVNULL)

    # https://docs.unity3d.com/Manual/CommandLineArguments.html
    self.proc = subprocess.Popen([bin,
                                  *(['-logfile'] if self.log_unity else []),
                                  *(['-batchmode', '-nographics'] if self.batchmode else []),
                                  '-screen-width {}'.format(self.w),
                                  '-screen-height {}'.format(self.h),
                                  ],
                                 env=env,
                                 stdout=output_redirect,
                                 stderr=output_redirect,
                                 universal_newlines=True,
                                 )

    def poll():
      while not self.proc.poll():
        sleep(1)
      logger.debug(f'Unity returned with {self.proc.returncode}')

    threading.Thread(target=poll, daemon=True).start()

    # wait until connection with simulator process
    timeout = 20
    for i in range(timeout * 10):
      if self.proc.poll():
        logger.debug('simulator died')
        break

      try:
        self.soc.connect((host, port))
        self.soc.settimeout(20 * 60)  # 20 minutes
        self.connected = True
        break
      except ConnectionRefusedError as e:
        if i == timeout * 10 - 1:
          print(e)

      sleep(.1)

    if not self.connected:
      raise ConnectionRefusedError('Connection with simulator could not be established.')

  def _reset(self):
    if not self.configured:
      self.conf()

    if self.restart:
      self.disconnect()
      self.restart = False

    if not self.connected:
      self.connect()

    else:
      self.send(np.zeros(2), reset=True)

    # skip first observation from simulator because it's faulty
    # TODO: fix first observation in simulator
    self.receive()
    self.send(np.zeros(2), reset=False)

    state, frame = self.receive()

    return state, frame

  def receive(self):
    pixel_buffer_size = 0 if self.batchmode else self.w * self.h * 4
    buffer_size = self.sd * 4 + pixel_buffer_size
    # receive data from simulator process
    data_in = b""
    while len(data_in) < buffer_size:
      chunk = self.soc.recv(min(1024, buffer_size - len(data_in)))
      data_in += chunk

    # Checking data points are not None, if yes parse them.
    if self.wp is None:
      with open(os.path.join(self.sim_path, 'sim_Data', 'waypoints_SimpleTerrain.txt')) as f:
        try:
          wp = json.load(f)
          self.wp = np.array([[e['x'], e['y'], e['z']] for e in wp])
          logger.debug(str(self.wp))
        except json.JSONDecodeError:
          self.wp = None

    # Read the number of float sent by the C# side. It's the first number
    # sd = int(np.frombuffer(data_in, np.float32, 1, 0))
    # assert sd == self.sd, f'State dimension expected: {self.sd}, received: {sd}'

    state = np.frombuffer(data_in, np.float32, self.sd, 0)

    if self.batchmode:
      frame = None
    else:
      # convert frame pixel data into a numpy array of shape [width, height, 3]
      frame = np.frombuffer(data_in, np.uint8, -1, self.sd * 4)
      # logger.debug(str(len(frame)))
      frame = np.reshape(frame, [self.w, self.h, 4])
      frame = frame[::-1, :, :3]

    self.last_frame = frame
    self.last_state = state

    return state, frame

  def send(self, action, reset=False):
    a = np.concatenate((action, [1. if reset else 0.]))
    a = np.array(a, dtype=np.float32)
    assert a.shape == (self.ad + 1,)

    data_out = a.tobytes()
    self.soc.sendall(data_out)

  def disconnect(self):
    if self.proc:
      self.proc.kill()
    if self.soc:
      self.soc.close()
    self.connected = False

  def _close(self):
    logger.debug('close')
    if self.proc:
      self.proc.kill()
    if self.soc:
      self.soc.close()
    if self.logfile:
      self.logfile.close()

  def _render(self, mode='human', close=False):
    if mode == 'rgb_array':
      return self.last_frame  # return RGB frame suitable for video
    elif mode is 'human':
      pass  # we do that anyway
    else:
      super()._render(mode, close)  # just raise an exception


def get_free_port(host):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind((host, 0))
  port = sock.getsockname()[1]
  sock.close()
  return port