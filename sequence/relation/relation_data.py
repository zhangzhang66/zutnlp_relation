#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : relation_data
# @Author   : ZhiYi Zhang
# @Time     : 2020/10/22 20:16

import random
from torch.utils.data import Dataset
import json
import nltk
import os
import six
from six.moves import cPickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from common.data.common_data_loader import CommonDataLoader
from common.util.utils import timeit
from sequence.relation.relation_dataset import REDataset, Loader, train_collate, Train_data, Data, dev_test_collate
from sequence.relation.util import build_char, build_tags, read_json, build_word_dict, build_rel2id, build_labels


class RelationDataLoader(CommonDataLoader):

    def __init__(self, data_config):
        super(RelationDataLoader, self).__init__(data_config)
        self.__build_field()
        self._load_data()
        self.__build_vocab()
        self.__build_iterator()

        pass

    def __build_field(self):
        build_char(self._config.data.char2id_path)
        build_tags(self._config.data.pos2id_path)
        train_data = read_json(self._config.data.train_path)
        dev_data = read_json(self._config.data.train_path)
        build_word_dict(train_data + dev_data, self._config.data.word_file_path)
        build_rel2id(train_data, self._config.data.rel2id_path)
        build_labels(self._config.data.label2id_path)
        REDataset(self._config).prepare()

    @timeit
    def _load_data(self):
        self.train_data = Train_data(self._config)

        self.dev_data = Data(self._config.data.root, 'dev')
        self.test_data = Data(self._config.data.root, 'test')

    def __build_vocab(self, *dataset):
        """
        :param dataset: train_data, valid_data, test_data
        :return: text_vocab, tag_vocab
        """
        pass

    def __build_iterator(self):
        self.dev_iter = iter(DataLoader(self.dev_data, batch_size=1, collate_fn=dev_test_collate, shuffle=False))
        pass

    def load_train(self):
        train_data = []
        for x in iter(
                DataLoader(self.train_data, batch_size=self._config.learn.train_batch_size, collate_fn=train_collate,
                           shuffle=True)):
            train_data.append(Loader(self._config).get_batch_train(self._config.learn.train_batch_size))
        return train_data

        pass

    def load_valid(self):
        return self.dev_iter
        pass

    def load_test(self):
        return iter(
            DataLoader(self.test_data, batch_size=self._config.learn.eval_batch_size, collate_fn=dev_test_collate,
                       shuffle=False))
        pass


if __name__ == '__main__':
    config_file = 'relation_config.yml'
    import dynamic_yaml

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(config_file, mode='r', encoding='UTF-8') as f:
        config = dynamic_yaml.load(f)
    data_loader = RelationDataLoader(config)
    train_dataloader = data_loader.load_train()
    for dict_input in tqdm(train_dataloader):
        text = dict_input.sents[0]
        text_len = dict_input.example_l
    pass
