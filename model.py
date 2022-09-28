#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/9/25
# project = model

import torch.nn as nn
import torch
from transformers import BertModel


class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased', num_labels=2)
        self.l1 = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert_layer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return torch.nn.functional.softmax(self.l1(output.pooler_output), dim=1)
