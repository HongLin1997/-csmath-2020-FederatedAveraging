# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:34:15 2020

@author: admin
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
 
colors = np.array(sns.color_palette(sns.color_palette('Paired', 12)))
seed = 0

if __name__ == '__main__':
    datasets = ['mnist', ] #'cifar', 
    models = ['softmax', ]
    epochs = ['200']
    num_users = ['100'] #100
    fracs = ['0.1', '0.2', '0.3', '0.4', '0.5',
             '0.6', '0.7', '0.8', '0.9', '1.0'] #0.5
    noniid = ['2']
    local_ep = ['5'] #[i for i in range(10)]
    local_bs = ['10']
    unb = 0.0 #2.0
    ita = 0.01
    for dataset in datasets:
        loss_curves = dict()
        global_acc_curves = dict()
        local_acc_curves = dict()
        for m in models:
            for e in epochs:
                for n in num_users:
                    for f in fracs:
                        for lep in local_ep:
                            for lbs in local_bs:
                                for iid in noniid:
                                    loss_train = pickle.load(open('./save/seed{}_fed{}_loss_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}ita{}.png'.format(seed, 'fedavg', dataset, m, e, n, f, iid, unb, lep, lbs, ita),'rb'))
                                    global_acc_tests = pickle.load(open('./save/seed{}_fed{}_gacc_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}ita{}.png'.format(seed, 'fedavg', dataset, m, e, n, f, iid, unb, lep, lbs, ita),'rb'))
                                    local_acc_test = pickle.load(open('./save/seed{}_fed{}_lacc_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}ita{}.png'.format(seed, 'fedavg', dataset, m, e, n, f, iid, unb, lep, lbs, ita),'rb'))
                                    loss_curves['{}_r{}_K{}C{}_E{}B{}'.format(m, e, n, f, lep, lbs)] = loss_train
                                    global_acc_curves['{}_r{}_K{}C{}_E{}B{}'.format(m, e, n, f, lep, lbs)] = global_acc_tests
                                    local_acc_curves['{}_r{}_K{}C{}_E{}B{}'.format(m, e, n, f, lep, lbs)] = local_acc_test
                                    
        plt.figure(figsize=(20,10))
        for i, (label, c) in enumerate(loss_curves.items()):
            plt.plot(range(len(c)), c, label= label, color=colors[i])
        
        plt.ylabel('train loss')
        plt.xlabel('communication round')
        plt.legend()
        plt.title(dataset)
        plt.savefig('./save/fed_loss_{}_iid{}unb{}.png'.format(dataset, noniid, unb))
        
        plt.figure(figsize=(20,10))
        for i, (label, c) in enumerate(global_acc_curves.items()):
            plt.plot(range(len(c)), c, label= label, color=colors[i])
        
        plt.ylabel('global test accuracy')
        plt.xlabel('communication round')
        plt.legend()
        plt.title(dataset)
        plt.savefig('./save/fed_gacc_{}_iid{}unb{}.png'.format(dataset, noniid, unb))
        
        plt.figure(figsize=(20,10))
        for i, (label, c) in enumerate(local_acc_curves.items()):
            plt.plot(range(len(c)), c, label= label, color=colors[i])
        
        plt.ylabel('local test accuracy')
        plt.xlabel('communication round')
        plt.legend()
        plt.title(dataset)
        plt.savefig('./save/fed_lacc_{}_iid{}unb{}.png'.format(dataset, noniid, unb))
        
    