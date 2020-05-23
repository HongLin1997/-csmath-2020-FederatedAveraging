# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:34:15 2020

@author: admin
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, train=True, local_model=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if train:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), 
                                        batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), 
                                        batch_size=1000, shuffle=True)
        self.local_model = local_model
        
    def test(self, net):
        net.eval()
        # testing
        test_loss = 0
        correct = 0
        
        for idx, (data, target) in enumerate(self.ldr_train):
            if self.args.gpu != -1:
                data, target = data.to(self.args.device), target.to(self.args.device)
            net_outputs = net(data)
            # sum up batch loss
            test_loss += F.cross_entropy(net_outputs[-1], target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = net_outputs[-1].data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            
        test_loss /= len(self.ldr_train.dataset)
        accuracy = 100 * int(correct) / len(self.ldr_train.dataset)
        return (accuracy, len(self.ldr_train.dataset)), test_loss

    def train(self, net, lr):
        
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)
        
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                net.zero_grad()
                net_outputs = net(images)
                loss = self.loss_func(net_outputs[-1], labels)

                loss.backward()
                optimizer.step()
                
                #optimizer_discriminator.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

