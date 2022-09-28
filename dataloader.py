#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/9/25
# project = dataloader
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class SSTDataset(Dataset):
    def __init__(self,filename):
        self.df=pd.read_csv(filename,delimiter='\t')
        # self.tokenizer=BertTokenizer.from_pretrained('/Users/zhangzhenhua/Code/python/TJU/huggingface/bert')
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence=self.df.loc[index,'sentence']
        label=self.df.loc[index,'label']
        token=self.tokenizer(sentence,padding='max_length',truncation=True,max_length=32,return_tensors='pt')
        return token,torch.tensor(label)
