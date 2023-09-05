from pandas.core.common import not_none
from rl_model.config import args
from rl_model.config.mode import Mode
from rl_model.config.utils import set_global_seed, hide_warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ShowPlot:
    def __init__(self,args):
        self.logdir = args.path.split(",")
        self.smoothing = args.smoothing
        self.x_axis = args.x_axis
        self.len_data = 971
        
    def load_data(self,path):
        data_frame = pd.read_csv(path+"monitor_train.csv", index_col=None)
        metric = data_frame['right_action']

        if self.x_axis == 'epochs' or self.x_axis == 'epochs_std':
            
            y = None
            y_mean = metric.groupby(np.arange(len(metric.index))//self.len_data, axis=0).mean()
            y_std = metric.groupby(np.arange(len(metric.index))//self.len_data, axis=0).std()

            x = range(len(y_mean))
            x_label = 'Number of epochs'
            window = 1

        else:
            timesteps = data_frame["timesteps"]

            y = metric
            y_mean = None
            y_std = None

            window = self.smoothing

            if self.x_axis == 'timesteps':
                x = np.cumsum(timesteps)
                x_label = 'Number of timesteps'
            
            elif self.x_axis == 'episodes':
                x = range(len(timesteps))
                x_label = 'Number of episodes'

            else:
                pass

        
        return x, y, y_mean, y_std, x_label, window

    def __call__(self):
      """
      plot the results

      :param log_folder: (str) the save location of the results to plot
      :param title: (str) the title of the task to plot
      """    
        
      title='Learning Curve'
      
      fig = plt.figure(title,figsize=(40,10))
      
      for path in self.logdir: 
          model_name = path.split("/")[-2]

          x, y, y_mean, y_std, x_label, window = self.load_data(path)

          if self.x_axis == 'epochs_std':
              plt.plot(x, y_mean,label=model_name)
              plt.fill_between(x,y_mean-y_std,y_mean+y_std,alpha=.1)
          elif self.x_axis == 'epochs':
              plt.plot(x, y_mean,label=model_name)
          else:
              weights = np.repeat(1.0, window) / window
              y = np.convolve(y, weights, 'valid')
              x = x[len(x) - len(y):]
              plt.plot(x, y, label=model_name)

      plt.xlabel(x_label)
      plt.ylabel('Accuracy')
      plt.title(title + " Smoothed")
      plt.legend(loc="upper left")
      plt.show()
      #plt.savefig(self.logdir+'best_model.png')


if __name__ == "__main__":
    # define console functions
    plot_training = ShowPlot(args.config(mode=Mode.PLOT))
    plot_training()    


    
