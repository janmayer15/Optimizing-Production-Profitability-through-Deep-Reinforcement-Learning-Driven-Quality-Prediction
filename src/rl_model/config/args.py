import argparse
from rl_model.config.mode import Mode

def config(mode):
      parser = argparse.ArgumentParser(description = 'Define variables for training the model.')

      parser.add_argument('--path', required=True, type=str, help='Enter the path, where the model should be trained.')
      parser.add_argument('--study_name', type=str, help="Set study name for hyperparamater tuning.")

      if mode == Mode.TRAIN or mode == Mode.HYPERTRAIN:
          parser.add_argument('--timesteps', default=100, type=int, help="Set the number of timesteps for the training.")
          parser.add_argument('--standard_callback', action='store_true',help="Call it for standard callback (verbose) from stable baselines")
          parser.add_argument('--custom_callback', action='store_true',help="Call it for custom callback.")
          parser.add_argument('--train_eval_n_episodes', default=100, type=int, help="After how many episodes train data should be evaluated.")
          parser.add_argument('--cross_validation', default=None, type=str, help="Call it if you want to crossvalidate on a metric (f1, acc, sens, spez, gm).")
          parser.add_argument('--save_every_x_steps', default=0, type=int, help="After how many steps a model should be saved.")
          parser.add_argument('--early_stopping',  default=0, type=int, help="Define after how many non imporving evaluations the training has to be stopped.")
          parser.add_argument('--train_eval_freq',  default=500, type=int, help="After how many episodes model should be evaluated on train data.")
          parser.add_argument('--val_eval_freq',  default=5000, type=int, help="After how many episodes model should be evaluated on val data.")

      # set network parameters
      if mode == Mode.TRAIN:
          parser.add_argument('--double_q', action='store_true', help="If true DDQN will be applied, otherwise DQN.")
          parser.add_argument('--layer_normalization', action='store_true', help="If true layers of the network will be normalized.")
          parser.add_argument('--dueling', action='store_true', help="After how many episodes train data should be evaluated.")
          parser.add_argument('--buffer_size_replay', default=50000, type=int, help="Memory length for replay.")
          parser.add_argument('--prioritized_replay', action='store_true', help="Prioritized replay.")

      # set environment parameters
      if mode == Mode.TRAIN or mode == Mode.EVAL:
          parser.add_argument('--pos_reward_good_wafer', default='1.0', type=str, help="Reward when the agent predicts the good wafer right/wrong.")
          parser.add_argument('--neg_reward_good_wafer', default='1.0', type=str, help="Punishment when the agent predicts the good wafer right/wrong.")
          parser.add_argument('--pos_reward_bad_wafer', default='1.0', type=str, help="Reward when the agent predicts the bad wafer right/wrong.")
          parser.add_argument('--neg_reward_bad_wafer', default='1.0', type=str, help="Punishment when the agent predicts the bad wafer right/wrong.")
          parser.add_argument('--discount_wait', default='0.001', type=str, help="Negative discount factor in the reward function for the action waiting.")
          parser.add_argument('--train_best_hypermodel', action='store_true', help="Train best hypermodel.")

      if mode == Mode.EVAL:
          parser.add_argument('--evaluate_on_test', action='store_true',help="Call it when evaluating only on test data.")
          parser.add_argument('--evaluate_on_train', action='store_true',help="Call it when evaluating only on training data.")
          parser.add_argument('--evaluation_summary', action='store_true',help="Print evaluation summary.")
          parser.add_argument('--confusion_matrix', action='store_true',help="Print confusion matrix.")


      # settings for hyperparameter tuning
      if mode == Mode.HYPERTRAIN:
          parser.add_argument('--optimize_again', action='store_true', help="Load Study and optimize again.")
          parser.add_argument('--trials', default=5 , type=int, help="Number of trials for hyperparameter tuning.")
          parser.add_argument('--timeout', default=None , type=int, help="Max. time of running.")
          parser.add_argument('--enqueue_trial', default=None, type=int, help="Enqueue failed trial.")

      if mode == Mode.PLOT:
          parser.add_argument('--smoothing', default=1, type=int, help="Enter the smoothing window to calculate the moving average when plotting the training curve.")
          parser.add_argument('--x_axis', default='timesteps', type=str, help="Print timesteps or episodes on the x axis (timesteps, episodes, epochs, epochs_std).")

      return parser.parse_args()

   