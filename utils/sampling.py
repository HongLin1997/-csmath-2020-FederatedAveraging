#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import random, math, os, torch

    
def iid(dataset, num_users, train=True, unbalance=None, props=None):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    if unbalance:
        mean = 0 #unbalance[0]
        sigma = unbalance
        if type(props)!=type(None):
            o_props = props
            pass
        else:
            props = np.random.lognormal(mean, sigma, (num_users))
            props = props/props.sum()
            o_props = props
            
        #print(props.sum())
        #num_items = int(len(dataset)/num_users)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, max(int(props[i]*len(dataset)), 1), replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i]) # exclude what were selected
        return dict_users, o_props
    
    else:
        num_items = int(len(dataset)/num_users)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i]) # exclude what were selected
        return dict_users, None


def mnist_noniid(dataset, num_users, train=True):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    if train:
        labels = dataset.train_labels.numpy()
    else:
        labels = dataset.test_labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def mnist_noniid_unequal(dataset, num_users, train=True):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    if train:
        labels = dataset.train_labels.numpy()
    else:
        labels = dataset.test_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def noniid_dist(dataset, num_users, class_per_device, 
                max_data_per_device=None, noniidness=0, equal_dist=False, 
                train=True, users_classes=None, classes_devives=None, unbalance=None, props=None ):
    '''
    noniidness: percentage of non-iid data per device
    max_data_per_device: max amount of data per device
    '''
    try:
        if train:
            classes = len(set(np.array(dataset.train_labels)))
        else:
            classes = len(set(np.array(dataset.test_labels)))
    except:
        classes = len(set(np.array(dataset.targets)))
            
    if class_per_device:   # Use classes per device
        if class_per_device > classes:
            raise OverflowError("Class per device is larger than number of classes")

        if equal_dist:
            raise NotImplementedError("Class per device can only be used with unequal distributions")
        else:
            
            # Distribute class numbers to devices
            if train:
                current_classs = 0
                users_classes = [[] for i in range(num_users)]  # Classes dictionary for devices
                classes_devives = [[] for i in range(classes)]  # Devices in each class
    
                for i in range(num_users):
                    next_current_class = (current_classs+class_per_device)%classes
                    if next_current_class > current_classs:
                        users_classes[i] = np.arange(current_classs, next_current_class)
                    else:
                        users_classes[i] = np.append(
                            np.arange(current_classs, classes),
                            np.arange(0, next_current_class)
                        )
                        
                    for j in users_classes[i]:
                        classes_devives[j].append(i)
                        
                    current_classs = next_current_class
            else:
                users_classes = users_classes
                classes_devives = classes_devives
                
            # unbalancing
            if unbalance:
                dataset_group = []
                for i in range(10):
                    try:
                        if train:
                            idx = dataset.train_labels==i
                        else:
                            idx = dataset.test_labels==i
                    except:
                        idx = np.array(dataset.targets)==i
                    dataset_group.append(dataset.data[idx])
                
                print([len(v) for v in dataset_group])
                
                mean = 0 #unbalance[0]
                sigma = unbalance
                if type(props)!=type(None):
                    o_props = props
                    props = np.array([[[len(v)]] for v in dataset_group])*props/np.sum(props,(1,2), keepdims=True)
                    pass
                else:
                    props = np.random.lognormal(mean, sigma, (classes, int(num_users/classes), class_per_device))
                    o_props = props
                    props = np.array([[[len(v)]] for v in dataset_group])*props/np.sum(props,(1,2), keepdims=True)
                    
            # Combine indexes and labels for sorting
            try:
                if train:
                    idxs_labels = np.vstack((np.arange(len(dataset)), np.array(dataset.train_labels)))    
                else:
                    idxs_labels = np.vstack((np.arange(len(dataset)), np.array(dataset.test_labels)))    
            except:
                idxs_labels = np.vstack((np.arange(len(dataset)), np.array(dataset.targets)))    
                
            idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

            users_idxs = [[] for i in range(num_users)]  # Index dictionary for devices

            if unbalance:
                print('unbalancing')
                current_idx = 0
                for i in range(classes):
                    if not len(classes_devives[i]):
                        continue
                    device_user_index = classes_devives[i]
                    send_to_device = 0
                    
                    class_data_index = np.where(idxs_labels[1, :]==i)[0]
                    for xx, device_user in enumerate(device_user_index):
                        num_samples = int(props[i, xx//class_per_device, xx%class_per_device])
                        if num_samples <=0:
                            num_samples = 1
                        choice = np.random.choice(class_data_index, num_samples, replace=False)
                        class_data_index = np.array(list(set(class_data_index)-set(choice)))
                        users_idxs[device_user] += list(idxs_labels[0, choice])
                        
            else:    
                current_idx = 0
                for i in range(classes):
                    if not len(classes_devives[i]):
                        continue
    
                    send_to_device = 0
                    for j in range(current_idx, len(idxs_labels[0])):
                        if idxs_labels[1, j] != i:
                            current_idx = j
                            break
                        users_idxs[classes_devives[i][send_to_device]].append(idxs_labels[0, j])
                        send_to_device = (send_to_device+1)%len(classes_devives[i])

                if max_data_per_device:
                    for i in range(num_users):
                        users_idxs[i] = users_idxs[i][0:max_data_per_device]
                
            dict_users = {i: np.array(users_idxs[i], dtype='int64') for i in range(len(users_idxs))}

            return dict_users, users_classes, classes_devives, (o_props if unbalance else None)

    else:   # Use non-IIDness
        if equal_dist:
            data_per_device = math.floor(len(dataset)/num_users)
            users_idxs = [[] for i in range(num_users)]  # Index dictionary for devices

            # Combine indexes and labels for sorting
            try:
                if train:
                    idxs_labels = np.vstack((np.arange(len(dataset)), np.array(dataset.train_labels)))    
                else:
                    idxs_labels = np.vstack((np.arange(len(dataset)), np.array(dataset.test_labels)))    
            except:
                idxs_labels = np.vstack((np.arange(len(dataset)), np.array(dataset.targets)))    
             
            idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
            idxs = idxs_labels[0, :].tolist()
            niid_data_per_device = int(data_per_device*noniidness/100)

            # Distribute non-IID data
            for i in range(num_users):
                users_idxs[i] = idxs[i*niid_data_per_device:(i+1)*niid_data_per_device]

            # Still have some data
            if num_users*niid_data_per_device < len(dataset):
                # Filter distributed data
                idxs = idxs[num_users*niid_data_per_device:]
                # Randomize data after sorting
                random.shuffle(idxs)

                remaining_data_per_device = data_per_device-niid_data_per_device

                # Distribute IID data
                for i in range(num_users):
                    users_idxs[i].extend(idxs[i*remaining_data_per_device:(i+1)*remaining_data_per_device])
                
            if max_data_per_device:
                for i in range(num_users):
                    users_idxs[i] = users_idxs[i][0:max_data_per_device]
                
            dict_users = {i: np.array(users_idxs[i], dtype='int64') for i in range(len(users_idxs))}

            return dict_users, users_classes, classes_devives, None
        else:
            # Max data per device
            max = math.floor(len(dataset)/num_users)
            # Each device get [0.2*max, max) amount of data
            data_per_device = [int(random.uniform(max/5, max)) for i in range(num_users)]

            users_idxs = [[] for i in range(num_users)]  # Index dictionary for devices

            # Combine indexes and labels for sorting
            try:
                if train:
                    idxs_labels = np.vstack((np.arange(len(dataset)), np.array(dataset.train_labels)))    
                else:
                    idxs_labels = np.vstack((np.arange(len(dataset)), np.array(dataset.test_labels)))    
            except:
                idxs_labels = np.vstack((np.arange(len(dataset)), np.array(dataset.targets)))    
             
            #idxs_labels = np.vstack((np.arange(len(dataset)), np.array(dataset.train_labels.numpy())))    
            idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
            idxs = idxs_labels[0, :].tolist()

            niid_data_per_device = [int(data_per_device[i]*noniidness/100) for i in range(num_users)]

            current_idx = 0
            # Distribute non-IID data
            for i in range(num_users):
                users_idxs[i] = idxs[current_idx:current_idx+niid_data_per_device[i]]
                current_idx += niid_data_per_device[i]

            # Filter distributed data
            idxs = idxs[current_idx:]
            # Randomize data after sorting
            random.shuffle(idxs)

            remaining_data_per_device = [data_per_device[i]-niid_data_per_device[i] for i in range(num_users)]

            current_idx = 0
            # Distribute IID data
            for i in range(num_users):
                users_idxs[i].extend(idxs[current_idx:current_idx+remaining_data_per_device[i]])
                current_idx += remaining_data_per_device[i]

            if max_data_per_device:
                for i in range(num_users):
                    users_idxs[i] = users_idxs[i][0:max_data_per_device]

            dict_users = {i: np.array(users_idxs[i], dtype='int64') for i in range(len(users_idxs))}

            return dict_users, users_classes, classes_devives, None
            

def get_datasets(args):
    # load datasets
    if args.dataset == 'mnist':
        img_size = torch.Size([args.num_channels, 28, 28])
        
        trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
        if os.path.exists('./data/mnist/processed/test.pt'):
            download = False
        else:
            download = True
        dataset_train = datasets.MNIST(
                root='./data/mnist/',
                train=True,
                download=download,
                transform=trans_mnist
            )

        dataset_test = datasets.MNIST(
                root='./data/mnist/',
                train=False,
                download=download,
                transform=trans_mnist
            )
        
    elif args.dataset == 'cifar':
        img_size = torch.Size([3, 32, 32])

        trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])
        trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])
    
        dataset_train = datasets.CIFAR10(
                root='./data/cifar',
                train=True,
                download=True,
                transform=trans_cifar10_train
            )
    
        dataset_test = datasets.CIFAR10(
                root='./data/cifar',
                train=False,
                download=True,
                transform=trans_cifar10_val
            )
    else:
        exit('Error: unrecognized dataset')
        
        
    return dataset_train, dataset_test, img_size

def data_split(args):
    # load datasets
    dataset_train, dataset_test, img_size = get_datasets(args)
    users_classes = [[j for j in range( args.num_classes)] for i in range( args.num_users)]
    # sample users
    if args.noniid==0:
        dict_users, props = iid(dataset_train, args.num_users, 
                                unbalance = args.unbalance)
        test_dict_users,_ = iid(dataset_test, args.num_users, train=False, 
                                unbalance = args.unbalance, props = props)
    elif args.noniid==1:
        print('args.noniid=',args.noniid, 
              'args.class_per_device=', args.class_per_device,
              'args.unbalance=', args.unbalance)
        dict_users, users_classes, classes_devives, props = noniid_dist(dataset_train, 
                                                                        args.num_users, 
                                                                        class_per_device=args.class_per_device, 
                                                                        unbalance = args.unbalance)
        test_dict_users, _, _, _ = noniid_dist(dataset_test, args.num_users, 
                                               class_per_device=args.class_per_device,
                                               train=False, users_classes = users_classes, 
                                               classes_devives = classes_devives, 
                                               unbalance = args.unbalance, props = props)
    elif args.noniid==3:
        dict_users, _, _, _ = noniid_dist(dataset_train, args.num_users,
                                          class_per_device=0, 
                                          max_data_per_device=args.max_data_per_device,
                                          noniidness=args.noniidness, equal_dist=True)
        test_dict_users, _, _, _ = noniid_dist(dataset_test, args.num_users,
                                       class_per_device=0, 
                                       max_data_per_device=args.max_data_per_device,
                                       noniidness=args.noniidness, equal_dist=True, 
                                       train=False)
    elif args.noniid==4:
        dict_users, _, _ , _= noniid_dist(dataset_train, args.num_users,
                                          class_per_device=0, 
                                          max_data_per_device=args.max_data_per_device,
                                          noniidness=args.noniidness, equal_dist=False)
        test_dict_users, _, _, _  = noniid_dist(dataset_test, args.num_users, 
                                        class_per_device=0, 
                                        max_data_per_device=args.max_data_per_device,
                                        noniidness=args.noniidness, equal_dist=False,
                                        train=False)
    
    else:
        exit('Error: unrecognized args.noniid')
        
    p_k = []
    x=[]
    for i in dict_users.values():
        x.append(len(i))
    print('train data partition')
    print('sum:',np.sum(np.array(x)))
    print('mean:',np.mean(np.array(x)))
    print('std:',np.std(np.array(x)))
    
    p_k = np.array(x)/sum(x)
    
    x=[]
    for i in test_dict_users.values():
        x.append(len(i))
    print('test data partition')
    print('sum:',np.sum(np.array(x)))
    print('mean:',np.mean(np.array(x)))
    print('std:',np.std(np.array(x)))
    
    return dataset_train, dict_users, dataset_test, test_dict_users, p_k, img_size, users_classes

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
