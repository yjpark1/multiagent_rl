# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:55:33 2018

@author: yj-wn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette="bright")

flatui = ["#e41a1c", "#f781bf", "#4daf4a", "#984ea3", "#377eb8", "#ff7f00"]

sns.set_palette(flatui)

scenarios = ['simple_spread', 'simple_speaker_listener',
             'fullobs_collect_treasure', 'multi_speaker_listener']

index = 'env_partial'

loc_dir = {
            'MADDPG': 'Models/' + index + '/maddpg',
            'BiCNet': 'Models/' + index + '/BICnet',
            'MAAC': 'Models/' + index + '/MAAC',
            'MADR': 'Models/' + index + '/proposed+gumbel',
            'MADR+AMLAPN': 'Models/' + index + '/proposed+gumbel+model',
            'MADDPG+AMLAPN': 'Models/' + index + '/maddpg+amlapn+env_partial',
           }

N = 40000

def readResult(root, scenario):
    # root = 'Models/BICnet'
    # scenario = scenarios[0]
    files = os.listdir(root)
    files = [x for x in files if 'history' in x]
    files = [x for x in files if not 'test' in x]
    sc_files = [x for x in files if scenario in x]
    rewards = []
    for f in sc_files:
        path = os.path.join(root, f)
        reward = np.load(path)
        reward = reward['reward_episodes']
        reward = reward[:N]
        rewards.append(reward)
    rewards = np.stack(rewards)
    return rewards


def Plot(sc_index):
    result = {
              'MADDPG': 0,
              'BiCNet': 0,
              'MAAC': 0,
              'MADR': 0,
              'MADR+AMLAPN': 0,
              'MADDPG+AMLAPN': 0,
             }

    for mth, root in loc_dir.items():
        result[mth] = readResult(root, scenario=scenarios[sc_index])


    # Smoothing CI plot
    data_smt = {'Method': [],
                'Episode': [],
                'seed': [],
                'Reward': [],}

    for k, re in result.items():
        for seed, reward in enumerate(re):
            reward = reward.tolist()
            reward = pd.DataFrame(reward).rolling(window=50).mean()
            reward = reward.dropna().values
            reward = reward.flatten()

            N = len(reward)
            episode = np.arange(N).tolist()
            method = np.repeat(k, N).tolist()
            seed = np.repeat(seed, N).tolist()
            reward = reward.tolist()

            data_smt['Method'] += method
            data_smt['seed'] += seed
            data_smt['Episode'] += episode
            data_smt['Reward'] += reward

    a_smt = pd.DataFrame(data_smt)
    plt.subplots(figsize=(5, 3.75))

    if sc_index == 0:
        legend = 'brief'
    else:
        legend = False

    sns.lineplot(x="Episode", y="Reward",
                 hue="Method", data=a_smt, n_boot=10,
                 ci=95, legend=legend)
    plt.savefig(scenarios[sc_index] + '_' + index + '_new.png',
                dpi=300, bbox_inches='tight')
    plt.show()

#######
for sc_index in range(4):
    Plot(sc_index)


'''
#a = 5/4
#print(4 * a)
#print(3 * a)
import seaborn as sns
from matplotlib import pyplot as plt
sns.set(style="whitegrid")
tips = sns.load_dataset("tips")

plt.subplots(figsize=(5, 3.75))
sns.palplot(sns.color_palette("Set1", 8))
plt.show()

plt.subplots(figsize=(5, 3.75))
sns.palplot(sns.color_palette("bright", 8))
plt.show()
'''