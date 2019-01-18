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
sns.set(palette="Set2")

scenarios = ['simple_spread', 'simple_reference', 'simple_speaker_listener',
             'fullobs_collect_treasure', 'multi_speaker_listener']

loc_dir = {
            'MADDPG': 'Models/maddpg',
            'BiCNet': 'Models/BICnet',
            'MAAC': 'Models/MAAC',
            'Proposed': 'Models/proposed+gumbel',
            'Proposed+model': 'Models/proposed+gumbel+model',
           }

N = 40000

def readResult(root, scenario):
    # root = 'Models/BICnet'
    # scenario = scenarios[0]
    files = os.listdir(root)
    files = [x for x in files if 'history' in x]
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
              'Proposed': 0,
              'Proposed+model': 0,
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
            reward = pd.DataFrame(reward).rolling(window=100).mean()
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
    plt.subplots(figsize=(8,6))
    sns.lineplot(x="Episode", y="Reward",
                 hue="Method", data=a_smt, n_boot=10,
                 ci=95)
    plt.show()
    plt.savefig(scenarios[sc_index] + '.png', dpi=100)


#######
Plot(sc_index=0)
Plot(sc_index=3)
Plot(sc_index=4)

for sc_index in range(5):
    Plot(sc_index)