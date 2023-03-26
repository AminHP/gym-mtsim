# stable-baselines3, Add Gymnasium support: 
# https://github.com/DLR-RM/stable-baselines3/pull/1327
# pip install git+https://github.com/DLR-RM/stable-baselines3@feat/gymnasium-support

import os
import sys
sys.path.append('./gym-mtsim') # Optionally set path correctly (if not installed via 'pip' -> ModuleNotFoundError)

import gymnasium as gym
import gym_mtsim

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# RL Algorithms: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
from stable_baselines3 import A2C, PPO

# Implemented in SB3 Contrib 
# install SB3 Contrib + gymnasium-support
# pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib@feat/gymnasium-support
from sb3_contrib import RecurrentPPO, TRPO, MaskablePPO

from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import random
import torch


def print_stats(start_balance, reward_over_episodes):
    """print Balance / Reward / Returns at end """

    avg = np.mean(reward_over_episodes)
    min = np.min(reward_over_episodes)
    max = np.max(reward_over_episodes)

    print (f'Balance at start     : {start_balance:>10.3f}')
    print (f'Min. Reward          : {min:>10.3f}')
    print (f'Avg. Reward          : {avg:>10.3f}')
    print (f'Max. Reward          : {max:>10.3f}')

    end_balance = start_balance + avg
    returns = (end_balance - start_balance) / start_balance * 100
    out_returns = ''
    if returns > 0: out_returns = '+'
    out_returns += f'{returns:.3f} %' 

    print (f'Avg. Balance at end  : {end_balance:>10.3f} ({out_returns})')

    return min, avg, max, returns

# ProgressBarCallback for model.learn()
class ProgressBarCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose: int = 1):
        super(ProgressBarCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.progress_bar = tqdm(total=self.model._total_timesteps, desc="model.learn()")
        pass

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.progress_bar.update(self.check_freq)
            pass

        return True
    
    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.progress_bar.close()
        pass

# TRAINING + TEST
# =========================================================
def train_test_model(model, env, seed, total_num_episodes, total_learning_timesteps=10_000):
    """ if model=None then execute 'Random actions' """

    #reproduce training and test
    print ('-' * 80)
    obs = env.reset(seed=seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    vec_env = None

    if model is not None:
        print(f'model {type(model)}')
        print(f'policy {type(model.policy)}')
        #print(f'model.learn(): {total_learning_timesteps} timesteps ...')

        #custom callback for 'progress_bar'
        model.learn(total_timesteps=total_learning_timesteps, callback=ProgressBarCallback(100))
        #model.learn(total_timesteps=total_learning_timesteps, progress_bar=True)
        # ImportError: You must install tqdm and rich in order to use the progress bar callback. 
        # It is included if you install stable-baselines with the extra packages: `pip install stable-baselines3[extra]`

        vec_env = model.get_env()
        obs = vec_env.reset()
    else:
        print ("RANDOM actions")

    reward_over_episodes = []

    tbar = tqdm(range(total_num_episodes))

    for episode in tbar:
        
        if vec_env: 
            obs = vec_env.reset()
        else:
            obs, info = env.reset()

        total_reward = 0
        done = False
        while not done:

            if model is not None:
                action, _states = model.predict(obs)
                obs, reward, done, info = vec_env.step(action)
            else: #random
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated    

            total_reward += reward
            if done: break
        
        reward_over_episodes.append(total_reward)

        if episode % 10 == 0:
            avg_reward = np.mean(reward_over_episodes)
            tbar.set_description(f'Episode: {episode}, Avg. Reward: {avg_reward:.3f}')
            tbar.update()

    tbar.close()
    avg_reward = np.mean(reward_over_episodes)
    
    return reward_over_episodes

# -------------------------------------------------------------------------------------
# INIT Env.
# -------------------------------------------------------------------------------------
#env_name = 'forex-hedge-v0'
env_name = 'stocks-hedge-v0'
#env_name = 'crypto-hedge-v0'
#env_name = 'mixed-hedge-v0'

#env_name = 'forex-unhedge-v0'
#env_name = 'stocks-unhedge-v0'
#env_name = 'crypto-unhedge-v0'
#env_name = 'mixed-unhedge-v0'

start_balance = 50_000.00

env = gym.make(env_name)
env.original_simulator.balance = start_balance
env.original_simulator.equity = start_balance

seed = 42 #random seed
total_num_episodes = 1000

print ("env_name                 :", env_name)
print ("seed                     :", seed)


# INIT matplotlib
plot_settings = {}
plot_data = {'x': [i for i in range(1, total_num_episodes + 1)]}

# Random actions
model = None 
total_learning_timesteps = 0
rewards = train_test_model(model, env, seed, total_num_episodes, total_learning_timesteps)
min, avg, max, returns = print_stats(start_balance, rewards)
label = f'Random actions ({returns:.2f} %)'
plot_data['rnd_rewards'] = rewards
plot_settings['rnd_rewards'] = {'label': label}


learning_timesteps_list_in_K = [1, 3, 5]
#learning_timesteps_list_in_K = [50, 250, 500]
#learning_timesteps_list_in_K = [500, 1000, 3000, 5000]

# RL Algorithms: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
#model_class_list = [A2C, PPO]
model_class_list = [A2C, PPO, RecurrentPPO, TRPO]

for timesteps in learning_timesteps_list_in_K:

    total_learning_timesteps = timesteps * 1000
    step_key = f'{timesteps}K'

    for model_class in model_class_list:
        
        policy_dict = model_class.policy_aliases
        # MultiInputPolicy or MultiInputLstmPolicy
        policy = policy_dict.get('MultiInputPolicy')
        if policy is None: policy = policy_dict.get('MultiInputLstmPolicy')
        #print ('policy:', policy, 'model_class:', model_class)

        model = model_class(policy, env, verbose=0)
        class_name = type(model).__qualname__

        plot_key = f'{class_name}_rewards_'+step_key
        rewards = train_test_model(model, env, seed, total_num_episodes, total_learning_timesteps)
        min, avg, max, returns = print_stats(start_balance, rewards)
        label = f'{class_name} - {step_key} ({returns:.2f} %)'
        plot_data[plot_key] = rewards
        plot_settings[plot_key] = {'label': label}



data = pd.DataFrame(plot_data)

sns.set_style('whitegrid')
plt.figure(figsize=(8, 6))

for key in plot_data:

    if key == 'x': continue
    label = plot_settings[key]['label']
    line = plt.plot('x', key, data=data, linewidth=1, label=label)

plt.xlabel('episode')
plt.ylabel('reward')
plt.title('Random vs. SB3 Agents')
plt.legend()
plt.show()


"""

Results for:
- learning_timesteps_list_in_K = [500, 1000, 3000, 5000]
- model_class_list = [A2C, PPO]

--------------------------------------------------------------------------------
RANDOM actions
Balance at start     :  50000.000
Min. Reward          : -18089.240
Avg. Reward          :  -1679.978
Max. Reward          :  15524.900
Avg. Balance at end  :  48320.022 (-3.360 %)

--------------------------------------------------------------------------------
model <class 'stable_baselines3.a2c.a2c.A2C'>
model.learn(): 500000 timesteps ...
Balance at start     :  50000.000
Min. Reward          :   2972.891
Avg. Reward          :   3308.991
Max. Reward          :   3609.601
Avg. Balance at end  :  53308.991 (+6.618 %)

--------------------------------------------------------------------------------
model <class 'stable_baselines3.ppo.ppo.PPO'>
model.learn(): 500000 timesteps ...
Balance at start     :  50000.000
Min. Reward          :    726.590
Avg. Reward          :   1156.107
Max. Reward          :   1607.530
Avg. Balance at end  :  51156.107 (+2.312 %)
--------------------------------------------------------------------------------

model <class 'stable_baselines3.a2c.a2c.A2C'>
model.learn(): 1000000 timesteps ...
Balance at start     :  50000.000
Min. Reward          :    943.150
Avg. Reward          :   1167.195
Max. Reward          :   1398.490
Avg. Balance at end  :  51167.195 (+2.334 %)
--------------------------------------------------------------------------------

model <class 'stable_baselines3.ppo.ppo.PPO'>
model.learn(): 1000000 timesteps ...
Balance at start     :  50000.000
Min. Reward          :   1182.670
Avg. Reward          :   1669.154
Max. Reward          :   2267.010
Avg. Balance at end  :  51669.154 (+3.338 %)
--------------------------------------------------------------------------------

model <class 'stable_baselines3.a2c.a2c.A2C'>
model.learn(): 3000000 timesteps ...
Balance at start     :  50000.000
Min. Reward          :   1619.480
Avg. Reward          :   1833.296
Max. Reward          :   2024.420
Avg. Balance at end  :  51833.296 (+3.667 %)
--------------------------------------------------------------------------------

model <class 'stable_baselines3.ppo.ppo.PPO'>
model.learn(): 3000000 timesteps ...
Balance at start     :  50000.000
Min. Reward          :   2534.469
Avg. Reward          :   2973.081
Max. Reward          :   3547.970
Avg. Balance at end  :  52973.081 (+5.946 %)
--------------------------------------------------------------------------------

model <class 'stable_baselines3.a2c.a2c.A2C'>
model.learn(): 5000000 timesteps ...
Balance at start     :  50000.000
Min. Reward          :  13704.384
Avg. Reward          :  13793.400
Max. Reward          :  13900.946
Avg. Balance at end  :  63793.400 (+27.587 %)
--------------------------------------------------------------------------------

model <class 'stable_baselines3.ppo.ppo.PPO'>
model.learn(): 5000000 timesteps ...
Balance at start     :  50000.000
Min. Reward          :   3159.979
Avg. Reward          :   3600.301
Max. Reward          :   4110.480
Avg. Balance at end  :  53600.301 (+7.201 %)

"""