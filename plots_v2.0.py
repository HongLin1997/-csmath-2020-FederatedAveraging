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
seed = 42
flag = 'E'

if __name__ == '__main__':
    datasets = ['mnist', ] #'cifar', 
    models = ['softmax', ]
    epochs = ['200']
    num_users = ['100'] #100
    fracs = [0.1] # ['0.1', '0.2', '0.3', '0.4', '0.5','0.6', '0.7', '0.8', '0.9', '1.0'] # 
    noniid = ['2']
    local_ep = [i+1 for i in range(10)] # ['5'] # 
    local_bs = ['10']
    unb = 2.0 #2.0
    ita = 0.01
    max1,max2,max3 = [],[],[]
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
                                    loss_train = pickle.load(open('./save/seed{}_fed{}_loss_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}ita{}.pkl'.format(seed, 'fedavg', dataset, m, e, n, f, iid, unb, lep, lbs, ita),'rb'))
                                    global_acc_tests = pickle.load(open('./save/seed{}_fed{}_gacc_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}ita{}.pkl'.format(seed, 'fedavg', dataset, m, e, n, f, iid, unb, lep, lbs, ita),'rb'))
                                    local_acc_test = pickle.load(open('./save/seed{}_fed{}_lacc_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}ita{}.pkl'.format(seed, 'fedavg', dataset, m, e, n, f, iid, unb, lep, lbs, ita),'rb'))
                                    loss_curves['{}_r{}_K{}C{}_E{}B{}'.format(m, e, n, f, lep, lbs)] = loss_train
                                    global_acc_curves['{}_r{}_K{}C{}_E{}B{}'.format(m, e, n, f, lep, lbs)] = global_acc_tests
                                    local_acc_curves['{}_r{}_K{}C{}_E{}B{}'.format(m, e, n, f, lep, lbs)] = local_acc_test
                                    max1.append(sum(loss_train))
                                    max2.append(sum(global_acc_tests))
                                    max3.append(sum(local_acc_test))
        loss_curves = list(loss_curves.items())
        plt.figure(figsize=(20,10))
        for i in np.argsort(max1):
            (label, c) = loss_curves[i]
            plt.plot(range(len(c)), c, label= label, color=colors[i])
        
        plt.ylabel('train loss',fontsize=20)
        plt.xlabel('communication round',fontsize=20)
        plt.legend(fontsize=20)
        plt.title(dataset,fontsize=20)
        plt.savefig('./save/flag{}_fed_loss_{}_iid{}unb{}.png'.format(flag, dataset, iid[0], unb))
        
        global_acc_curves = list(global_acc_curves.items())
        plt.figure(figsize=(20,10))
        for i in np.argsort(max2)[::-1]:
            (label, c) = global_acc_curves[i]
            plt.plot(range(len(c)), c, label= label, color=colors[i])
        
        plt.ylabel('global test accuracy',fontsize=20)
        plt.xlabel('communication round',fontsize=20)
        plt.legend(fontsize=20)
        plt.title(dataset,fontsize=20)
        plt.savefig('./save/flag{}_fed_gacc_{}_iid{}unb{}.png'.format(flag, dataset, iid[0], unb))
        
        local_acc_curves = list(local_acc_curves.items())
        plt.figure(figsize=(20,10))
        for i in np.argsort(max3)[::-1]:
            (label, c) = local_acc_curves[i]
            plt.plot(range(len(c)), c, label= label, color=colors[i])
        
        plt.ylabel('local test accuracy',fontsize=20)
        plt.xlabel('communication round',fontsize=20)
        plt.legend(fontsize=20)
        plt.title(dataset,fontsize=20)
        plt.savefig('./save/flag{}_fed_lacc_{}_iid{}unb{}.png'.format(flag, dataset, iid[0], unb))
        
    