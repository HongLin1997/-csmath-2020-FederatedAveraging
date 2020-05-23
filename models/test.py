# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:34:15 2020

@author: admin
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.Update import LocalUpdate
    
def test_img(net_g, datatest, args, centers=None):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=1000)
    datas = []
    for idx, (data, target) in enumerate(data_loader):
        if type(datas) == list:
            datas = data
        else:
            datas = torch.cat((datas,data),0)
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        net_outputs = net_g(data)
                     
        # sum up batch loss
        test_loss += F.cross_entropy(net_outputs[-1], target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = net_outputs[-1].data.max(1, keepdim=True)[1]
                    
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss
    

def test_img_local(net_g, dataset_test, dict_users, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    
    acc_locals, local_data_sizes, loss_locals = [], [], []
    idxs_users = range(args.num_users)
    for idx in idxs_users:
        local = LocalUpdate(args=args, dataset=dataset_test, idxs=dict_users[idx], train=False)
        (acc, local_data_size), loss = local.test(net=net_g.to(args.device))
        acc_locals.append(acc)
        local_data_sizes.append(local_data_size)
        loss_locals.append(loss)
    test_loss = sum([i*j for i, j in zip(local_data_sizes, loss_locals)])/sum(local_data_sizes)
    correct = sum([i*j for i, j in zip(local_data_sizes, acc_locals)])
    accuracy = correct/sum(local_data_sizes)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, sum(local_data_sizes), accuracy))
    
    return acc_locals, accuracy, test_loss


