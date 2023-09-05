from data import Data
from env import CustomEnv

'''buffer_size=50000, 
            exploration_fraction=0.1, 
            exploration_final_eps=0.02, 
            exploration_initial_eps=1.0, 
            train_freq=1, 
            batch_size=32, 
            learning_starts=1000, 
            target_network_update_freq=500, 
            prioritized_replay=False, 
            prioritized_replay_alpha=0.6, 
            prioritized_replay_beta0=0.4, 
            prioritized_replay_beta_iters=None, 
            prioritized_replay_eps=1e-06, 
            param_noise=False, 
            n_cpu_tf_sess=None,
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.00005, 1.),
            '''

import lightgbm as lgb
SEED = 42
np.random.seed(SEED)

class TrainHyperparams:
    def __init__(self):
        self.x_train, self.x_test, self.y_train, self.y_test = Data().prepare_data()
        self.env_train = CustomEnv(self.x_train,self.y_train,random_index=True,flatten=True)
        self.timesteps = 2000


    def objective(self,trial):
       
        model_params = {
            'learning_rate': trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
            'double_q': trial.suggest_categorical("double_q", [False,True]),
            'param_noise': trial.suggest_categorical("param_noise", [False,True]),
            'batch_size': trial.suggest_categorical("batch_size", [32, 64, 128])
        }

        Mlp = trial.suggest_categorical("classifier", [MlpPolicy, LnMlpPolicy])

        model = DQN(Mlp, self.env, verbose=0, **model_params)
        model.learn(self.timesteps)
        mean_reward, _ = evaluate_policy(model,self.env, n_eval_episodes=100)

        return mean_reward
        


if __name__ == "__main__":
    #Train().run()
    train = TrainHyperparams()
    '''study = optuna.create_study()
    study.optimize(train.optimize_agent, n_trials=100, n_jobs=4)'''

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),)
    study.optimize(train.objective, n_trials=100, timeout=600)
