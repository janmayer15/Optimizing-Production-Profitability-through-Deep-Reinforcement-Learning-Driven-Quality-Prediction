import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import logging

def set_global_seed(seed=100):
  os.environ['PYTHONHASHSEED']=str(seed) # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
  # python
  random.seed(seed) # 2. Set `python` built-in pseudo-random generator at a fixed value
  # numpy
  np.random.seed(seed) # 3. Set `numpy` pseudo-random generator at a fixed value
  # tensorflow
  tf.compat.v1.set_random_seed(seed) # 4. Set `tensorflow` pseudo-random generator at a fixed value


def set_pandas_options():
  pd.set_option('display.max_columns', 20)
  pd.set_option('display.max_rows', 20)
  pd.set_option('display.width', 2000)


def hide_warnings():
  tf.get_logger().setLevel('INFO')
  tf.autograph.set_verbosity(0)
  tf.get_logger().setLevel(logging.ERROR)

from json import JSONEncoder
class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__ 