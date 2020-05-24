# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:34:15 2020

@author: admin
"""


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy, os, pickle, sys
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.sampling import data_split
from utils.options import args_parser
from utils.tools import init_random_seed, adjust_learning_rate

from models.Update import LocalUpdate
from models.Nets import MLP, CNN, CNNCifar, SoftmaxClassifier
from models.Fed import FedAvg
from models.test import test_img, test_img_local

            
if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args)
    
    filename_nn = './save/seed{}_net-{}_glob_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}.pt'.format(args.manual_seed, 'naive', args.dataset, args.model.replace('_prog1','').replace('_dist','').replace('_prox',''), args.epochs, args.num_users, args.frac, args.class_per_device, args.unbalance, 1, 100)
    if args.dataset=='mnist' or (args.dataset=='cifar' and args.lr == 0.1):
        filename = './save/seed{}_net-{}_glob_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}.pt'.format(args.manual_seed, 'fedavg', args.dataset, args.model, args.epochs, args.num_users, args.frac, args.class_per_device, args.unbalance, args.local_ep, args.local_bs)
    else:
        filename = './save/seed{}_net-{}_glob_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}ita{}.pt'.format(args.manual_seed, 'fedavg', args.dataset, args.model, args.epochs, args.num_users, args.frac, args.class_per_device, args.unbalance, args.local_ep, args.local_bs, args.lr)
    
    if os.path.exists(filename):
        print('Already trained model and saved in {}'.format(filename))
        sys.exit(0)
    
    if args.manual_seed:
        init_random_seed(args.manual_seed)
    
    # load dataset and split users
    dataset_train, dict_users, dataset_test, test_dict_users, p_k, img_size, users_classes = data_split(args)
       
########################## centralized learning #############################
    if args.centralized and (not os.path.exists(filename_nn)):
        print('centralized learning...')
        # build model for centralized learning
        if (args.model == 'cnn' ) and args.dataset == 'mnist':
            net_glob = CNN(args=args).to(args.device)
        elif (args.model == 'cnn')  and args.dataset == 'cifar':
            net_glob = CNNCifar(args=args).to(args.device)
        elif args.model == 'mlp':
            len_in = 1
            for x in img_size:
                len_in *= x
            net_glob = MLP(dim_in=len_in, dim_hidden=200, 
                           dim_out=args.num_classes).to(args.device)
        elif args.model == 'softmax':
            len_in = 1
            for x in img_size:
                len_in *= x
            net_glob = SoftmaxClassifier(dim_in=len_in,
                                         dim_out=args.num_classes).to(args.device)
        else:
            exit('Error: unrecognized model')
    
        print(net_glob)
        net_glob.train()
    
        # training
        train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
        optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
        
        list_loss = []
        global_acc_tests = []
        local_acc_tests = []
        net_glob.train()
        for epoch in range(args.epochs):
            batch_loss = []
            lr = args.lr * (args.decay_rate ** (epoch // args.per_epoch))
            adjust_learning_rate(optimizer, lr)
            
            for batch_idx, (data, target) in enumerate(train_loader):    
                data, target = data.to(args.device), target.to(args.device)
                optimizer.zero_grad()
                output = net_glob(data)
                loss = F.cross_entropy(output[-1], target)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())            
                
            loss_avg = sum(batch_loss)/len(batch_loss)
            print('\nTrain loss:', loss_avg)
            list_loss.append(loss_avg)
            
            # testing
            net_glob.eval()
            acc_test, _ = test_img(net_glob, dataset_test, args)
            local_acc_test, _, _ = test_img_local(net_glob, 
                                                  dataset_test, 
                                                  test_dict_users, args)
            global_acc_tests.append(acc_test)
            local_acc_tests.append(sum(local_acc_test)/len(local_acc_test))
            print('Round {:3d}, Average global accuracy {:.2f}'.format(epoch, acc_test))
            print('Round {:3d}, Average local accuracy {:.2f}'.format(epoch, sum(local_acc_test)/len(local_acc_test)))
            net_glob.train()
            
        # plot loss curve
        plt.figure()
        plt.plot(range(len(list_loss)), list_loss)
        plt.ylabel('train_loss')
        plt.savefig('./save/seed{}_naive_loss_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}.png'.format(args.manual_seed, args.dataset, args.model, args.epochs, args.num_users, args.frac, args.class_per_device, args.unbalance, 1, 100 ))
        pickle.dump(list_loss,
                    open('./save/seed{}_naive_loss_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}.pkl'.format(args.manual_seed, args.dataset, args.model, args.epochs, args.num_users, args.frac, args.class_per_device, args.unbalance, 1, 100),'wb'))
        
        # plot global test acc curve
        plt.figure()
        plt.plot(range(len(global_acc_tests)), global_acc_tests)
        plt.ylabel('global_acc_tests')
        plt.savefig('./save/seed{}_naive_gacc_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}.png'.format(args.manual_seed, args.dataset, args.model, args.epochs, args.num_users, args.frac, args.class_per_device, args.unbalance, 1, 100))
        pickle.dump(global_acc_tests,
                    open('./save/seed{}_naive_gacc_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}.pkl'.format(args.manual_seed, args.dataset, args.model, args.epochs, args.num_users, args.frac, args.class_per_device, args.unbalance, 1, 100),'wb'))
        
        # plot local test acc curve
        plt.figure()
        plt.plot(range(len(local_acc_tests)), local_acc_tests)
        plt.ylabel('local_acc_tests')
        plt.savefig('./save/seed{}_naive_lacc_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}.png'.format(args.manual_seed, args.dataset, args.model, args.epochs, args.num_users, args.frac, args.class_per_device, args.unbalance, 1, 100))
        pickle.dump(local_acc_tests,
                    open('./save/seed{}_naive_lacc_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}.pkl'.format(args.manual_seed, args.dataset, args.model, args.epochs, args.num_users, args.frac, args.class_per_device, args.unbalance, 1, 100),'wb'))
       
        # save model
        torch.save(net_glob.state_dict(), filename_nn)
        print("save model to: {}".format(filename_nn))

####################### federated learning ###############################
    print('federated learning...')
        
    # rebuild model for federated learning
    if (args.model == 'cnn' ) and args.dataset == 'mnist':
        net_glob = CNN(args=args).to(args.device)
    elif (args.model == 'cnn')  and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, 
                       dim_out=args.num_classes).to(args.device)
    elif args.model == 'softmax':
            len_in = 1
            for x in img_size:
                len_in *= x
            net_glob = SoftmaxClassifier(dim_in=len_in,
                                         dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    
    print(net_glob)
    net_glob.train()

    # training
    loss_train = []
    global_acc_tests = []
    local_acc_tests = []
    centers = None
    
    for iter in range(args.epochs):
        
        lr = args.lr * (args.decay_rate ** (iter // args.per_epoch))

        w_locals, local_centers, loss_locals = [], [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        p_k_selected = []
        p_k_unselected = []
        for idx in set(range(args.num_users))-set(idxs_users):
            p_k_unselected.append(p_k[idx])
        
        for idx in idxs_users:
            print(idx, 'client: ' , users_classes[idx])
            p_k_selected.append(p_k[idx])
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), 
                                                    lr=lr)
            
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        
        # update global weights
        w_glob = FedAvg(w_locals)
            
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        
        # testing
        net_glob.eval()
        acc_test, _ = test_img(net_glob, dataset_test, args)
        local_acc_test, _, _ = test_img_local(net_glob, 
                                              dataset_test, 
                                              test_dict_users, args)
        global_acc_tests.append(acc_test)
        local_acc_tests.append(sum(local_acc_test)/len(local_acc_test))
        print('Round {:3d}, Average global accuracy {:.2f}'.format(iter, acc_test))
        print('Round {:3d}, Average local accuracy {:.2f}'.format(iter, sum(local_acc_test)/len(local_acc_test)))
        
        net_glob.train()
        
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/seed{}_fed{}_loss_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}ita{}.png'.format(args.manual_seed, 'fedavg', args.dataset, args.model, args.epochs, args.num_users, args.frac, args.class_per_device, args.unbalance, args.local_ep, args.local_bs, args.lr))
    pickle.dump(loss_train,
                open('./save/seed{}_fed{}_loss_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}ita{}.pkl'.format(args.manual_seed, 'fedavg', args.dataset, args.model, args.epochs, args.num_users, args.frac, args.class_per_device, args.unbalance, args.local_ep, args.local_bs, args.lr),'wb'))
    
    # plot global test acc curve
    plt.figure()
    plt.plot(range(len(global_acc_tests)), global_acc_tests)
    plt.ylabel('global_acc_tests')
    plt.savefig('./save/seed{}_fed{}_gacc_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}ita{}.png'.format(args.manual_seed, 'fedavg', args.dataset, args.model, args.epochs, args.num_users, args.frac, args.class_per_device, args.unbalance, args.local_ep, args.local_bs, args.lr))
    pickle.dump(global_acc_tests,
                open('./save/seed{}_fed{}_gacc_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}ita{}.pkl'.format(args.manual_seed, 'fedavg', args.dataset, args.model, args.epochs, args.num_users, args.frac, args.class_per_device, args.unbalance, args.local_ep, args.local_bs, args.lr),'wb'))
    
    # plot local test acc curve
    plt.figure()
    plt.plot(range(len(local_acc_tests)), local_acc_tests)
    plt.ylabel('local_acc_tests')
    plt.savefig('./save/seed{}_fed{}_lacc_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}ita{}.png'.format(args.manual_seed, 'fedavg', args.dataset, args.model, args.epochs, args.num_users, args.frac, args.class_per_device, args.unbalance, args.local_ep, args.local_bs, args.lr))
    pickle.dump(local_acc_tests,
                open('./save/seed{}_fed{}_lacc_{}_{}_{}_K{}C{}_noniid{}unb{}_E{}B{}ita{}.pkl'.format(args.manual_seed, 'fedavg', args.dataset, args.model, args.epochs, args.num_users, args.frac, args.class_per_device, args.unbalance, args.local_ep, args.local_bs, args.lr),'wb'))
    
    torch.save(net_glob.state_dict(), filename)
    print("save model to: {}".format(filename))