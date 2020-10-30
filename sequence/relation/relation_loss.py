#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : relation_loss
# @Author   : ZhiYi Zhang
# @Time     : 2020/10/22 20:16
import torch
import torch.nn as nn

from sequence.relation.relation_model import device


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, input, target, mask):
        loss_total = -input.gather(dim=2, index=target.unsqueeze(2)).squeeze(2) * mask
        loss = torch.sum(loss_total)
        return loss


class LossWrapper(nn.Module):
    def __init__(self, Model, config):
        super(LossWrapper, self).__init__()
        self._config = config
        self.model = Model.to(device)
        self.criterion = CrossEntropy()

    def forward(self, dict_inputs: dict) -> dict:
        dict_inputs = [t.cuda() for t in dict_inputs[:-1]]
        sent, rel, label, pos, char, sen_len = dict_inputs
        mask = torch.zeros(sent.size()).to(device)
        for i in range(sent.size(0)):
            mask[i][:sen_len[i]] = 1

        mask2 = torch.where(label == 12, torch.ones_like(sent), torch.ones_like(sent) * 10).to(device)
        mask2 = mask2.float() * mask.float()
        predict, weight = self.model(sent, sen_len, rel, mask, pos, char)
        predict = torch.log_softmax(predict, dim=-1)

        loss = self.criterion(predict, label, mask2)
        return loss
