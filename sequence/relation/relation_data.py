#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : relation_data
# @Author   : ZhiYi Zhang
# @Time     : 2020/10/22 20:16
import torch
from torch.utils.data import DataLoader
from torchtext.data import Field, BucketIterator, NestedField
from torchtext.datasets import SequenceTaggingDataset

from common.data.common_data_loader import CommonDataLoader
from common.util.utils import timeit


def tokenizer(token):
    return [k for k in token]

class RelationDataLoader(CommonDataLoader):

    def __init__(self, data_config):
        super(RelationDataLoader, self).__init__(data_config)
        self.__build_field()
        self._load_data()
        pass

    def __build_field(self):

        pass