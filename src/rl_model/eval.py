import numpy as np
from sklearn.metrics import confusion_matrix

def model_evaluation(model,
                      env,
                      n_episodes: int = 300,
                      val_eval: bool = False,
                      total_eval: bool = False,
                      eval_table = None,
                      model_name = None):

    episode_rewards, episode_lengths = [],[]
    
    # for training callback evaluation only a few metrics are needed
    good_episode_lengths, bad_episode_lengths, y_true, y_pred = [],[],[],[]
    right_preds = 0

    for i in range(n_episodes):
        obs = env.reset()
        done, state = False, None          

        while not done:
            action, state = model.predict(obs)
            new_obs, _ , done, info = env.step(action)
            obs = new_obs

        if not total_eval:
          info = info[0]

        good_episode_length = 0
        bad_episode_length = 0      
        y_true.append(info['Label'])
        # if predicted action is 0  then it will be transformed to a bad wafer 
        if info['Action'] != 0:
          y_pred.append(info['Action'])
        else:
          y_pred.append(2)

        if info['Action']==1: # good
            good_episode_lengths.append(info['total timesteps'])
        if info['Action']==2: # bad
            bad_episode_lengths.append(info['total timesteps'])
            
            
        episode_rewards.append(info['total reward'])
        episode_lengths.append(info['total timesteps'])

        if info['Action']==info['Label']:
            right_preds += 1

    env.close()  

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_timesteps = np.mean(episode_lengths)
    std_timesteps = np.std(episode_lengths)
    acc = right_preds/n_episodes

    if total_eval:
        mean_timesteps_bad = np.mean(bad_episode_lengths)
        mean_timesteps_good = np.mean(good_episode_lengths)

    TN, FP, FN, TP = confusion_matrix(y_true,y_pred).ravel()
    
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP) 
    precision = TP/(TP+FP)
    f1 = 2*((precision*sensitivity)/(precision+sensitivity))
    gm = (sensitivity*specificity)**0.5
        #precision_neg = TN/(TN+FN)

    if val_eval:
        return mean_reward, std_reward, mean_timesteps, std_timesteps, acc, f1, sensitivity, specificity, gm

    if total_eval:
        return evaluation_table(eval_table,model_name, mean_reward, acc, sensitivity, precision, f1, specificity, gm, mean_timesteps, mean_timesteps_good, mean_timesteps_bad), TN, FP, FN, TP


def evaluation_table(df,model_name, mean_reward, acc, sensitivity, precision, f1 ,specificity, gm, mean_timesteps, mean_timesteps_good, mean_timesteps_bad):
    
    new_row = {'model':model_name, 
              'mean_reward':mean_reward, 
              'acc':acc, 
              'sensitivity':sensitivity, 
              'precision': precision,
              'f1':f1,
              'specificity':specificity,
              'gm':gm,
              'mean_timesteps':mean_timesteps,
              'mean_timesteps_good':mean_timesteps_good,
              'mean_timesteps_bad':mean_timesteps_bad}

    df = df.append(new_row,ignore_index=True)
   
    return df

def identify_best_models(df,metric):
    best_index = df[metric].nlargest(2).index.values
    best_models = list(df.iloc[best_index,0])
    new_colname = 'best_'+metric
    df[new_colname] = df['model'].map(lambda x: 'x' if x==best_models[0] or x==best_models[1] else '')
    
    return df





    