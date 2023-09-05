import gym
from stable_baselines.common.vec_env import DummyVecEnv
import numpy as np
import os
import pandas as pd
from rl_model.eval import model_evaluation
from stable_baselines.common.callbacks import BaseCallback, EventCallback, CallbackList
from stable_baselines.common.vec_env import VecEnv, sync_envs_normalization, DummyVecEnv
from typing import NamedTuple, Union, List, Dict, Any, Optional
import warnings


class TrainEvalCallback(EventCallback):

    def __init__(self, 
                 eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 100,
                 eval_freq: int = 500,
                 log_path: str = None,
                 save_models_freq: int = 0,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1):
        super(TrainEvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.rewards, self.right_actions, self.timesteps = [],[],[]
        self.save_models_freq = save_models_freq
        self.verbose = verbose
        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])
        assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"
        self.eval_env = eval_env
        self.log_path = log_path

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print()
            print("---------------------------------------------------------------------------------------")
            print("Model Evaluation on TRAINING DATA (last {} episodes)".format(self.n_eval_episodes))
            print("---------------------------------------------------------------------------------------")

    def _on_step(self) -> bool:
        
        if self.locals['done'] == True:
            self.rewards.append(self.locals['info']['total reward'])
            self.timesteps.append(self.locals['info']['total timesteps'])
            self.right_actions.append(self.locals['info']['Action'] == self.locals['info']['Label'])
        
        # TRAINING DATA EVAL
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            sync_envs_normalization(self.training_env, self.eval_env)
            episode = self.locals['num_episodes']

            if self.verbose > 0:
                last_x_rewards = self.rewards[-self.n_eval_episodes:]
                last_x_timesteps = self.timesteps[-self.n_eval_episodes:]
                last_x_right_actions = self.right_actions[-self.n_eval_episodes:]
                mean_reward = np.mean(last_x_rewards)
                std_reward = np.std(last_x_rewards)
                mean_ep_length = np.mean(last_x_timesteps)
                std_ep_length = np.std(last_x_timesteps)
                acc = np.mean(last_x_right_actions)
                training_state = self.n_calls/self.locals['total_timesteps']*100
                
                template = '[{:.2f}%/100%] train_steps {} (eps {}), mean_reward: {:.2f} +/- {:.2f}, acc: {:.2f}, mean_timesteps: {:.2f} +/- {:.2f}'
                print(template.format(training_state,self.n_calls,episode,mean_reward,std_reward,acc,mean_ep_length,std_ep_length))
        
        # SAVE MODEL AFTER EVERY X EPISODES
        if self.save_models_freq > 0 and self.n_calls % self.save_models_freq == 0:
            print("save model...")
            path = os.path.join(self.log_path, 'rl_model_{}_steps'.format(self.n_calls))
            self.model.save(path)
      
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # SAVE TRAINING HISTORY
        list_dict = {'timesteps':self.timesteps, 'reward':self.rewards, 'right_action':self.right_actions} 
        df = pd.DataFrame(list_dict) 
        df.to_csv(self.log_path+'monitor_train.csv', index=False) 


class ValEvalCallback(EventCallback):

    def __init__(self, 
                 eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 eval_freq: int = 50,
                 n_eval_episodes: int = None,
                 log_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 early_stop: int = 0,
                 verbose: int = 1,
                 eval_metric: str = 'f1'):
        super(ValEvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_metric = eval_metric
        self.metrics, self.timesteps = [],[]
        #self.f1_scores = {}
        self.best_metric = -np.inf
        self.counter = 0
        self.countinue_learning = True
        self.early_stop = early_stop
        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.log_path = log_path


    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type"
                          "{} != {}".format(self.training_env, self.eval_env))

    def _on_step(self) -> bool:
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            sync_envs_normalization(self.training_env, self.eval_env)
            mean_reward, std_reward, mean_timesteps, std_timesteps, acc, f1, sens, spez, gm = model_evaluation(model = self.model, 
                                                                                                          env = self.eval_env,
                                                                                                          n_episodes=self.n_eval_episodes,
                                                                                                          val_eval=True,
                                                                                                          total_eval = False)
            template = 'Evaluation on VALIDATION DATA - mean_reward: {:.2f} +/- {:.2f}, acc: {:.2f}, mean_timesteps: {:.2f} +/- {:.2f}, '+ self.eval_metric + ': {:.2f}'
            
            if self.eval_metric == 'f1':
                metric = f1
            elif self.eval_metric == 'acc':
                metric = acc
            elif self.eval_metric == 'sens':
                metric = sens
            elif self.eval_metric == 'spez':
                metric = spez
            elif self.eval_metric == 'gm':
                metric = gm
            else:
                metric = None

            print(template.format(mean_reward,std_reward,acc, mean_timesteps, std_timesteps, metric))
            
            self.metrics.append(metric)
            self.timesteps.append(self.n_calls)
            
            if metric > self.best_metric:
                self.counter = 0 # reset counter
                self.best_metric = metric
            else: 
                self.counter += 1

            if self.early_stop > 0:
              # self.early_stop is the number after how many non imporoving evaluations the training has to be stopped
              if self.counter >= self.early_stop:
                  print()
                  print("-- Early stopping! --")
                  self.countinue_learning = False

        return self.countinue_learning
        #return True
    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        list_dict = {'timesteps':self.timesteps, self.eval_metric:self.metrics} 
        df = pd.DataFrame(list_dict) 
        df.to_csv(self.log_path+'evaluation_val.csv', index=False) 

