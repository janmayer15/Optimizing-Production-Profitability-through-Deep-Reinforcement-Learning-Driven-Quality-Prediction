import gym
from gym import spaces
import numpy as np
import random
import pandas as pd 

from rl_model.config.utils import set_global_seed

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 sequences,
                 labels,
                 #random_index: bool = None,
                 evaluation: bool = False,
                 n_episodes: int = 300,
                 flatten: bool = None,
                 param_lambda: float = 0.001,
                 param_p: float = 1/3,
                 negative_reward_bad_wafer: float = 1.0,
                 positive_reward_bad_wafer: float = 1.0,
                 negative_reward_good_wafer: float = 1.0,
                 positive_reward_good_wafer: float = 1.0,
                 seed: int = None):
                 #path = None,
                 #monitor = False):
        super(CustomEnv, self).__init__()
        self.n_episodes = n_episodes
        self.current_index = 0
        self.sequences = sequences
        self.labels = labels
        self.sequence_i = self.sequences
        self.label_i = self.labels
        self.total_wafers = len(self.sequences)

        self.max_timesteps = 176
        self.features = 22
        self.flatten = flatten
        self.timestep = 1
        self.done = False
        self.reward = 0
        self.param_lambda = param_lambda
        self.param_p = param_p
        self.negative_reward_bad_wafer = negative_reward_bad_wafer
        self.positive_reward_bad_wafer = positive_reward_bad_wafer
        self.negative_reward_good_wafer = negative_reward_good_wafer
        self.positive_reward_good_wafer = positive_reward_good_wafer
        self.action_space = spaces.Discrete(3)
        self.SEED = seed
        set_global_seed(self.SEED)

        '''self.monitor = monitor
        if self.monitor == True:
          self.rewards, self.timesteps, self.right_action = [],[],[]'''
        #self.path = path

        if self.flatten == True:
          self.observation_space = spaces.Box(0, 2, shape=(3872,), dtype=np.float32)
        else:
          self.observation_space = spaces.Box(0, 2, shape=(176,22), dtype=np.float32)

    def step(self, action):
        current_timestep = self.timestep
        # if wait
        if action == 0:
          if self.timestep < self.max_timesteps: 
            self.reward += -(self.param_lambda * (self.timestep ** self.param_p))
            self.timestep += 1
            
          else:
            self.reward += -1
            #self.timestep = 1 
            self.done = True

        # if prediction = good
        if action == 1:
            # positve reward
            if self.label_i == 1:
                self.reward += self.positive_reward_good_wafer
            # negative reward
            else:
                self.reward += -self.negative_reward_good_wafer

            self.done = True

        # if prediction = bad
        if action == 2:
            # positve reward
            if self.label_i == 2:
                self.reward += self.positive_reward_bad_wafer
            # negative reward
            else:
                self.reward += -self.negative_reward_bad_wafer
            
            self.done = True

        seq = self.sequence_i[:self.timestep]
        
        if self.timestep < 176:
          zeros = np.zeros([self.max_timesteps-self.timestep,self.features])
          seq = np.concatenate((seq, zeros), axis=0)

        if self.flatten == True:
          obs = seq.flatten()
        else:
          obs = seq

        info = {'Action':action, 'Label':self.label_i ,'total reward':np.round(self.reward,4), 'total timesteps':current_timestep}
        
        '''if self.random_index == True:
          print(self.current_index)'''
        
        if self.done:
          self.current_index += 1
          
          '''if self.monitor:
            self.rewards.append(self.reward)
            self.timesteps.append(self.timestep)
            self.right_action.append(action==self.label_i)'''
          #print(self.current_index)
        

        return obs, self.reward, self.done, info

    def _next_observation(self):
        self.sequence_i = self.sequences[self.current_index]
        self.label_i = self.labels[self.current_index]
        
        obs = self.sequence_i[:self.timestep]
        
        if self.timestep < self.max_timesteps:
          zeros = np.zeros([self.max_timesteps-self.timestep,self.features])
          obs = np.concatenate((obs, zeros), axis=0)
       
        if self.flatten == True:
          observation = obs.flatten()
        else:
          observation = obs
          
        return observation

    def reset(self):
        if self.current_index == self.total_wafers:
          self.current_index = 0
      
        self.timestep = 1
        self.done = False
        self.reward = 0
        self.sequence_i = self.sequences
        self.label_i = self.labels      

        return self._next_observation()
    
    def close(self):
        self.current_index = 0
      