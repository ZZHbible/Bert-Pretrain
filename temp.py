#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/9/25
# project = temp
import pandas as pd


from transformers import BertTokenizer, BertModel


import torch

a= torch.tensor([[1,2,3],[2,3,4]],dtype=torch.float)
a=torch.nn.functional.softmax(a,dim=1)
print(a)
probs=torch.argmax(a,dim=1)
labels=torch.tensor([2,0])
# loss=torch.nn.CrossEntropyLoss()
# l=loss(a,labels)
# print(l)
acc = torch.sum((probs == labels))/len(probs)

print(acc)

