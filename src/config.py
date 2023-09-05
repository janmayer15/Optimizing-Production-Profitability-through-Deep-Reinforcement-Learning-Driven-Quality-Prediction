from rl_model.config.utils import set_global_seed, hide_warnings
import argparse

class Configuration:
    def __init__(self):
        self.SEED = 100

    def __call__(self):
      hide_warnings()
      set_global_seed(self.SEED)

    def get_seed(self):
      return self.SEED

if __name__ == "__main__":
    conf = Configuration()
    conf()


    
