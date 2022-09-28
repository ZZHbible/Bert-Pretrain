#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/9/18
# project = gpt-neox
import argparse
import time
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import SentimentClassifier
from dataloader import SSTDataset
from torch.utils.data.distributed import DistributedSampler  # 负责分布式dataloader创建，也就是实现上面提到的partition。

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 负责创建 args.local_rank 变量，并接受 torch.distributed.launch 注入的值
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()
# 每个进程根据自己的local_rank设置应该使用的GPU
torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)

# 初始化分布式环境，主要用来帮助进程间通信
torch.distributed.init_process_group(backend='nccl')

net=SentimentClassifier().to(device)
lr=5e-6
batch_size=16
criterion=nn.CrossEntropyLoss()
optim=torch.optim.Adam(net.parameters(),lr=lr)


# 只 master 进程做 logging，否则输出会很乱
if args.local_rank == 0:
    tb_writer = SummaryWriter(comment='ddp-training')

train_set=SSTDataset(filename='data/SST-2/train.tsv')
# 分布式数据集
train_sampler = DistributedSampler(train_set)
train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size)  # 注意这里的batch_size是每个GPU上的batch_size

val_set=SSTDataset(filename='data/SST-2/dev.tsv')
val_sampler=DistributedSampler(val_set)
val_loader=DataLoader(val_set,sampler=val_sampler,batch_size=batch_size)

for epoch in range(5):
    t=time.time()
    for index,(tokens,labels) in enumerate(train_loader):

        optim.zero_grad()
        labels=labels.to(device)
        input_ids=tokens['input_ids'].squeeze(1).to(device)
        attention_mask=tokens['attention_mask'].squeeze(1).to(device)
        token_type_ids=tokens['token_type_ids'].squeeze(1).to(device)
        logits=net(input_ids,attention_mask,token_type_ids) # tokens -> [B,max_len]
        loss=criterion(logits,labels)
        loss.backward()
        optim.step()

        if index%100==0:
            pred_labels = torch.argmax(logits, dim=1)  # 预测出的label
            acc = torch.sum(pred_labels == labels)/ len(pred_labels)  # acc
            print('epoch:{} pred_acc:{}'.format(epoch,acc))
    print("耗时:{}".format(time.time()-t))
    with torch.no_grad():
        mean_acc=0
        count=0
        for i,(tokens,labels) in enumerate(val_loader):
            labels=labels.to(device)
            input_ids=tokens['input_ids'].squeeze(1).to(device)
            attention_mask=tokens['attention_mask'].squeeze(1).to(device)
            token_type_ids=tokens['token_type_ids'].squeeze(1).to(device)
            logits=net(input_ids,attention_mask,token_type_ids) # tokens -> [B,max_len]
            count+=len(input_ids)
            pred_labels = torch.argmax(logits, dim=1)  # 预测出的label
            mean_acc+= torch.sum(pred_labels == labels)
        print("valid acc:"+str(mean_acc/count))


