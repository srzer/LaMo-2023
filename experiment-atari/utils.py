import os
import sys
import torch
import random
import logging
import numpy as np
import gym
import d4rl_atari
from gym.spaces import MultiDiscrete, Discrete, Box, Tuple
from collections import deque

class TorchDeque:
    def __init__(self, maxlen, device, dtype):
        self.maxlen = maxlen
        self.device = device
        self.dtype = dtype
        self.deque = deque(maxlen=maxlen)

    def append(self, array:torch.Tensor):
        if len(self.deque) == self.maxlen:
            self.deque.popleft() # Remove the oldest array
        self.deque.append(array.to(dtype=self.dtype, device=self.device)) # Add the new array

    def to_tensor(self):
        # Converts the deque of 1D arrays into a 2D array
        return torch.stack(list(self.deque), dim=1)

def config_logging(log_file="main.log"):
    date_format = '%Y-%m-%d %H:%M:%S'
    log_format = '%(asctime)s: [%(levelname)s]: %(message)s'
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Set up the FileHandler for logging to a file
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])

def get_log_dict(manager=None, num_seeds=0):
    log_keys = ['eval_steps', 'eval_returns', 'action_loss', 'rtg_target', 'perf_drop_train', 'perf_drop_finetune']

    if manager is None:
        return {key: [] for key in log_keys}
    else:
        return manager.dict({key: manager.list([[]] * num_seeds) for key in log_keys})

def set_seed_everywhere(env: gym.Env, seed=0):
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_space_shape(space, is_vector_env=False):
    if isinstance(space, Discrete):
        return space.n
    elif isinstance(space, MultiDiscrete):
        return space.nvec[0]
    elif isinstance(space, Box):
        space_shape = space.shape[1:] if is_vector_env else space.shape
        if len(space_shape) == 1:
            return space_shape[0]
        else:
            return space_shape  # image observation
    elif isinstance(space, Tuple):
        space_shape = get_space_shape(space[0], is_vector_env=False) if is_vector_env else sum([get_space_shape(space_i, is_vector_env=False) for space_i in space])
        return space_shape
    else:
        raise ValueError(f"Space not supported: {space}")
    
def write_to_dict(log_dict, key, value):
    log_dict[key][-1].append(value)

