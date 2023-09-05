# own modules
from multiprocessing import Value
from gym.core import Env

from numpy import rint
from numpy.lib.nanfunctions import nanargmin
from pandas._libs.hashtable import duplicated
from rl_model.data import Data
from rl_model.env import CustomEnv
from rl_model.callback import  TrainEvalCallback, ValEvalCallback
from rl_model.config.mode import Mode
from rl_model.config.utils import set_global_seed, set_pandas_options, hide_warnings
from rl_model.eval import model_evaluation, identify_best_models
import pandas as pd
import numpy as np
import os
import logging
import sys
import json
#import jsonpickle
from json import JSONEncoder

import math
from time import time
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.bench import Monitor
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, ConvertCallback

import optuna
from optuna.visualization import plot_param_importances

from sklearn.model_selection import StratifiedKFold


class RL_Experiment:
    
    def __init__(self,
                mode = None,
                args = None,
                seed = None):
        self.mode = mode
        self.args = args
        self.SEED = seed
        self.x_train, self.x_test, self.y_train, self.y_test = Data(seed=self.SEED).run()

        if not self.mode == Mode.HYPERTRAIN:
            if self.args.train_best_hypermodel:
                self.study = optuna.load_study(study_name=self.args.study_name, storage="sqlite:///{}.db".format(self.args.study_name))
                
                self.env_params = {'negative_reward_bad_wafer':self.study.best_trial.params['negative_reward_bad_wafer'],
                                  'positive_reward_bad_wafer':self.study.best_trial.params['positive_reward_bad_wafer'],
                                  'negative_reward_good_wafer':self.study.best_trial.params['negative_reward_good_wafer'],
                                  'positive_reward_good_wafer':self.study.best_trial.params['positive_reward_good_wafer'],
                                  'param_lambda':self.study.best_trial.params['param_lambda']}
            else:
                self.env_params = {'negative_reward_bad_wafer':float(self.args.neg_reward_bad_wafer),
                                  'positive_reward_bad_wafer':float(self.args.pos_reward_bad_wafer),
                                  'negative_reward_good_wafer':float(self.args.neg_reward_good_wafer),
                                  'positive_reward_good_wafer':float(self.args.pos_reward_good_wafer),
                                  'param_lambda':float(self.args.discount_wait)}
            
        if self.mode == Mode.EVAL:
            self.env_train = CustomEnv(self.x_train,self.y_train,**self.env_params,flatten=True,seed=self.SEED)
            self.env_test = CustomEnv(self.x_test,self.y_test,**self.env_params,flatten=True,seed=self.SEED)

            self.eval_episodes_traindata = 100
            self.column_names_eval = ["model", 
                                  "mean_reward", 
                                  "acc",
                                  "sensitivity",
                                  "precision",
                                  "f1",
                                  "specificity",
                                  "gm",
                                  "mean_timesteps",
                                  "mean_timesteps_good",
                                  "mean_timesteps_bad"]
        else:
            self.env_train = None
            self.env_val = None
            self.eval_episodes_traindata = self.args.train_eval_n_episodes #default 100
            self.eval_episodes_valdata = None # will be the length of validation set 
            self.split_cross_val = 4
            
            if (self.args.cross_validation or self.mode == Mode.HYPERTRAIN) and self.args.save_every_x_steps == 0:
                self.save_model_periodically = 5000 #set default
            else:
                self.save_model_periodically = self.args.save_every_x_steps
        
        
    def callback_train(self,verbose,path):
        callback = TrainEvalCallback(eval_env = self.env_train,
                                    n_eval_episodes = self.eval_episodes_traindata,
                                    eval_freq = self.args.train_eval_freq,
                                    verbose = verbose,
                                    log_path = path,
                                    save_models_freq = self.save_model_periodically)
        return callback

    def callback_val(self,verbose):
        callback = ValEvalCallback(eval_env = self.env_val,
                                    n_eval_episodes = self.eval_episodes_valdata,
                                    eval_freq = self.args.val_eval_freq,
                                    verbose = verbose,
                                    log_path = self.validation_path,
                                    early_stop = self.args.early_stopping,
                                    eval_metric = self.args.cross_validation)
        return callback


    def eval_crossvalidation_best_model(self):
        df = pd.read_csv(self.validation_path+"evaluation_val.csv", index_col=None)
        best_index = df[self.args.cross_validation].nlargest(1).index.values

        if len(best_index) == 0:
          best_metric = [0]
          best_trainingsteps = [df.iloc[-1,0]]
        else:
          best_metric = df.iloc[best_index,1].values
          best_trainingsteps = df.iloc[best_index,0].values

        return best_trainingsteps[0], best_metric[0]


    def load_model(self,logdir,name):
        model = DQN.load(logdir + name,env=self.env_train)
        return model

    # create model
    def model(self,network_params,model_params,dueling,verbose,callback):

        if self.mode == Mode.HYPERTRAIN or self.args.train_best_hypermodel:
            # Custom MLP policy
            class CustomDQNPolicy(FeedForwardPolicy):
                def __init__(self, *args, **kwargs):
                    super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                                      layers=[network_params['layer_1'], network_params['layer_2']],
                                                      layer_norm=network_params['layer_normalization'],
                                                      feature_extraction="mlp")
            model = DQN(CustomDQNPolicy,self.env_train,**model_params,policy_kwargs=dict(dueling=dueling),verbose=verbose,seed=self.SEED)

        else:
            model = DQN(network_params,self.env_train,**model_params,policy_kwargs=dict(dueling=dueling),verbose=verbose,seed=self.SEED)
        
        model.learn(total_timesteps=self.args.timesteps,callback=callback)
        
        return model 

    # NORMAL TRAINING
    def train_standard(self,model_params,network_params,dueling,env_params,verbose_standard,verbose_custom,path):
        
        self.env_train = CustomEnv(self.x_train,self.y_train,**env_params,flatten=True,seed=self.SEED)
        callback = self.callback_train(verbose_custom,path)
        
        start = time()
        model = self.model(network_params,model_params,dueling,verbose_standard,callback)
        end = time()
        print(end-start)

        if self.save_model_periodically == 0:
            model.save(os.path.join(path, 'rl_model_final'))
        
        return model

    # CROSSVALIDATION TRAINING
    def train_crossvalidation(self,model_params,network_params,dueling,env_params,path):

        skf = StratifiedKFold(n_splits=self.split_cross_val, shuffle=True, random_state=self.SEED)
        fold = 1
        folds, metrics, models = [],[],[]
        
        start1 = time()
        for train_index, val_index in skf.split(self.x_train, self.y_train):
            print()
            print("Split_"+str(fold))
            start = time()
            self.validation_path = path+"val_fold"+str(fold)+"/"
            os.makedirs(self.validation_path, exist_ok=True)

            x_train, y_train = self.x_train[train_index], self.y_train[train_index]
            x_val, y_val = self.x_train[val_index], self.y_train[val_index]  
            self.eval_episodes_valdata = len(y_val)

            self.env_train = CustomEnv(x_train,y_train,**env_params,flatten=True,seed=self.SEED)
            self.env_val = CustomEnv(x_val,y_val,**env_params,flatten=True,seed=self.SEED)

            callback_train = self.callback_train(1,self.validation_path)
            callback_val = self.callback_val(1)
            callback = CallbackList([callback_train, callback_val])

            model = self.model(network_params,model_params,dueling,0,callback)
      
            end = time()
            print()
            print(end-start)

            steps,metric = self.eval_crossvalidation_best_model()
            metrics.append(metric)
            models.append("rl_model_"+str(steps)+"_steps")
            folds.append(fold)
            
            fold += 1

        end1 = time()
        print("Time total")
        print(end1-start1)
        
        dict_list = {"fold":folds,"best_model":models,self.args.cross_validation:metrics}
        df_dict_list = pd.DataFrame(dict_list)
        df_dict_list.to_csv(path+'evaluation.csv', index=False)

        return np.mean(df_dict_list[self.args.cross_validation])
        

    # HYPERPARAMETER TRAINING
    def train_hyperparams(self,trial):
        current_training_folder = self.args.path + "training_trail_"+str(trial.number)+"/"
        os.makedirs(current_training_folder, exist_ok=True)

        # model params
        model_params = {
            'learning_rate': trial.suggest_categorical('learning_rate', [0.0001,0.001,0.01]),
            'gamma': trial.suggest_categorical('gamma', [0.9,0.95,0.99,0.995,0.999]),
            'double_q': trial.suggest_categorical('double_q', [False,True]),
            'param_noise': trial.suggest_categorical("param_noise", [False,True]),
            'learning_starts': trial.suggest_categorical('learning_starts', [500,1000,2000,3000]),
            'target_network_update_freq' : trial.suggest_categorical('target_network_update_freq', [500,1000,1500]),
            'exploration_fraction': trial.suggest_categorical('exploration_fraction', [0.0001,0.001,0.01]),
            'exploration_final_eps': trial.suggest_categorical('exploration_final_eps', [0.1,0.2,0.3]),
            'buffer_size': trial.suggest_categorical('buffer_size', [5000,10000,20000,30000,40000,50000]), 
            'batch_size': trial.suggest_categorical("batch_size", [32, 64, 128]),
            'prioritized_replay':trial.suggest_categorical('prioritized_replay', [False,True])
        }

        if model_params['prioritized_replay'] == True:
            model_params['prioritized_replay_alpha'] = trial.suggest_categorical('prioritized_replay_alpha', [0.6,0.65,0.7])
            model_params['prioritized_replay_beta0'] = trial.suggest_categorical('prioritized_replay_beta0', [0.4,0.5,0.6])
            

        network_params = {'layer_1': trial.suggest_categorical('layer_1', [32,64,128]),
                          'layer_2': trial.suggest_categorical('layer_2', [32,64,128]),
                          'layer_normalization': trial.suggest_categorical('layer_normalization', [True,False])}


        # environment params
        env_params = {'negative_reward_bad_wafer':trial.suggest_categorical('negative_reward_bad_wafer', [0.3,0.5,0.8,1.0]),
                      'positive_reward_bad_wafer':trial.suggest_categorical('positive_reward_bad_wafer', [0.3,0.5,0.8,1.0]),
                      'negative_reward_good_wafer':trial.suggest_categorical('negative_reward_good_wafer', [0.3,0.5,0.8,1.0]),
                      'positive_reward_good_wafer':trial.suggest_categorical('positive_reward_good_wafer', [0.3,0.5,0.8,1.0]),
                      'param_lambda': trial.suggest_categorical('param_lambda', [0.0001,0.001,0.01])}
        

        # dueling
        dueling = trial.suggest_categorical('dueling', [True,False])
        
      
        # check if this trails has parameter combinations that already exist
        study_name = self.args.study_name
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        df_study = study.trials_dataframe(attrs=("number", "value", "params"))
        df_noNAN = df_study.dropna(subset=['value'])
        df_currentTrial = df_study.iloc[trial.number,]
        df = df_noNAN.append(df_currentTrial).reset_index(drop=True)

        if df.iloc[: , 2:].duplicated().iloc[len(df)-1,]:
            bool_duplictes = np.array(df.iloc[: , 2:].duplicated(keep='last'))
            duplicate_trial_index = np.where(bool_duplictes==True)[0][0]
            value = df.iloc[duplicate_trial_index,1]

            if not math.isnan(value):
                return value
            else:
                return self.train_crossvalidation(model_params,network_params,dueling,env_params,current_training_folder)
        
        else: 
            return self.train_crossvalidation(model_params,network_params,dueling,env_params,current_training_folder)
        
    
    def __call__(self):
        set_global_seed(self.SEED)
        set_pandas_options()
        hide_warnings()

        # NORMAL TRAINING
        if self.mode == Mode.TRAIN:
            os.makedirs(self.args.path, exist_ok=True)

            env_params = self.env_params

            if self.args.train_best_hypermodel:
                model_params = {'learning_rate':self.study.best_trial.params['learning_rate'],
                                'gamma':self.study.best_trial.params['gamma'],
                                'double_q':self.study.best_trial.params['double_q'],
                                'param_noise':self.study.best_trial.params['param_noise'],
                                'learning_starts':self.study.best_trial.params['learning_starts'],
                                'target_network_update_freq':self.study.best_trial.params['target_network_update_freq'],
                                'exploration_fraction':self.study.best_trial.params['exploration_fraction'],
                                'exploration_final_eps':self.study.best_trial.params['exploration_final_eps'],
                                'buffer_size':self.study.best_trial.params['buffer_size'], 
                                'batch_size':self.study.best_trial.params['batch_size'],
                                'prioritized_replay':self.study.best_trial.params['prioritized_replay']}
                
                if model_params['prioritized_replay'] == True:
                    model_params['prioritized_replay_alpha'] = self.study.best_trial.params['prioritized_replay_alpha']
                    model_params['prioritized_replay_beta0'] = self.study.best_trial.params['prioritized_replay_beta0']
                  
                network_params = {'layer_1':self.study.best_trial.params['layer_1'],
                                  'layer_2':self.study.best_trial.params['layer_2'],
                                  'layer_normalization':self.study.best_trial.params['layer_normalization']}
                
                dueling = self.study.best_trial.params['dueling']

            else:
                # get model parameters from parsed arguments
                model_params = {'double_q':self.args.double_q,
                                'buffer_size':self.args.buffer_size_replay,
                                'prioritized_replay':self.args.prioritized_replay}

                dueling = self.args.dueling

                if self.args.layer_normalization:
                    network_params = LnMlpPolicy
                else:
                    network_params = MlpPolicy

            # CROSSVALIDATION
            if self.args.cross_validation:
                model = self.train_crossvalidation(model_params,network_params,dueling,env_params,self.args.path)
            
            # NORMAL TRAINING
            else:
                if self.args.standard_callback:
                    standard_verbose = 1
                    custom_verbose = 0
                else:
                    standard_verbose = 0
                    if self.args.custom_callback:
                        custom_verbose = 1
                    else:
                        custom_verbose = 0  
                
                model = self.train_standard(model_params,network_params,dueling,env_params,standard_verbose,custom_verbose,self.args.path)

        # HYPERPARAMETER TRAINING
        if self.mode == Mode.HYPERTRAIN:
            os.makedirs(self.args.path, exist_ok=True)
            #optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
            
            study_name = self.args.study_name
            storage_name = "sqlite:///{}.db".format(study_name)

            if self.args.optimize_again or self.args.enqueue_trial != None: 
                study = optuna.load_study(study_name=study_name, storage=storage_name)
            else:
                study = optuna.create_study(study_name=study_name,
                                        storage = storage_name,
                                        load_if_exists=True,
                                        direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=self.SEED))

            if not self.args.enqueue_trial == None:
                # put trial params here to rerun
                restart_trail = study.get_trials()[self.args.enqueue_trial].params
                study.enqueue_trial(restart_trail)
                study.optimize(self.train_hyperparams, n_trials=1)

            else:
                study.optimize(self.train_hyperparams, n_trials=self.args.trials, timeout=self.args.timeout)
            
            df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
            print()
            print("TRIAL SUMMARY")
            print(df)
            df.to_csv(self.args.path+'trail_summary.csv', index=False)
            print("Best Trial: ", study.best_trial)
            #plot_param_importances(study)


        # EVALUATE - evaluation summary table to identify the best model
        if self.mode == Mode.EVAL:
            
            if self.args.evaluate_on_test:
              eval_environment = {'Test_Data' : [self.env_test, self.x_test]} # evaluation only on test data
            elif self.args.evaluate_on_train:
              eval_environment = {'Train_Data' : [self.env_train, self.x_train]}  # evaluation only on train data
            else: # evaluation on train and test data
              eval_environment = {'Test_Data' : [self.env_test, self.x_test],
                                  'Train_Data' : [self.env_train, self.x_train]}

            for key, value in eval_environment.items():
                eval_table = pd.DataFrame(columns = self.column_names_eval)
                for file_x in os.listdir(self.args.path):
                    if file_x.endswith(".zip"):
                        model = self.load_model(logdir=self.args.path,name=file_x) 

                        # evaluate model
                        eval_table, TN, FP, FN, TP = model_evaluation(model = model,
                                                      env = value[0],
                                                      n_episodes=len(value[1]), 
                                                      val_eval = False,
                                                      total_eval = True,
                                                      eval_table=eval_table,
                                                      model_name=file_x) 

                # identify best model on follwing metrics
                for metric in ['mean_reward','f1','sensitivity','specificity','acc','gm']:
                    identify_best_models(eval_table,metric)
                
                if self.args.evaluation_summary:
                    print('Evaluation Summary on',key,'(',len(value[1]),'data points)')
                    print(eval_table)
                    print()
                    eval_table = pd.DataFrame(eval_table)
                    eval_table.to_csv(self.args.path+str(key)+'_evaluation.csv', index=False)

                if self.args.confusion_matrix:
                    print('Confusion matrix',key)
                    print('TN',TN,'FP',FP,'FN',FN,'TP', TP)
                    print()
        

