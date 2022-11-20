#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/9/25
# project = dataloader
import re

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

class SSTDataset(Dataset):
    def __init__(self, filename,train=True):
        self.df = pd.read_csv(filename)
        self.train=train
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence = self.df.loc[index, 'text']
        token = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
        if self.train:
            label = self.df.loc[index, 'target']
            return token, torch.tensor(label)
        return token

class TweetDataset(Dataset):
    def __init__(self, filename,train=True):
        self.df = pd.read_csv(filename)
        self.df.fillna(' ',inplace=True)
        self.train=train
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        # self.tokenizer=BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        def preprocess_text(document):
            # Remove all the special characters
            document = re.sub(r'\W', ' ', str(document))
            document = document.replace('_', ' ')

            # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

            # Substituting multiple spaces with single space
            document = re.sub(r'\s+', ' ', document, flags=re.I)

            # Converting to Lowercase
            document = document.lower()
            return document
        sentence = self.df.loc[index, 'text']
        sentence=preprocess_text(sentence)
        token = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=84, return_tensors='pt')
        if self.train:
            label = self.df.loc[index, 'target']
            return token, torch.tensor(label)
        return token