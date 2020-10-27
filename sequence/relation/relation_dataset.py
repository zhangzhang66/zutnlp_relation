# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/18 8:49
from torch.utils.data import Dataset
import sys
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
import random
import config
import json
import nltk
import os
import numpy as np
import six
from six.moves import cPickle

from sequence.relation.util import read_json


def pickle_load(f):
    if six.PY3:
        return cPickle.load(f, encoding='latin-1')
    else:
        return cPickle.load(f)


def pickle_dump(obj, f):
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)


class REDataset(object):

    def __init__(self, config, encoding="utf-8", **kwargs):
        self._config = config
        vocab = np.load(self._config.data.word_file_path)
        self.word2id = {j: i for i, j in enumerate(vocab)}
        self.id2word = {i: j for i, j in enumerate(vocab)}
        self.rel2id = json.load(open(self._config.data.rel2id_path, 'r'))
        self.label2id = json.load(open(self._config.data.label2id_path, 'r'))
        self.pos2id = json.load(open(self._config.data.pos2id_path, 'r'))
        self.char2id = json.load(open(self._config.data.char2id_path, 'r'))
        self.train_data = read_json(self._config.data.train_path)
        self.test_data = read_json(self._config.data.test_path)
        self.dev_data = read_json(self._config.data.valid_path)

    def prepare(self):
        print('loading data ...')
        train_pos_f, train_pos_l, train_neg_f, train_neg_l = self.process_train(self.train_data)
        with open(os.path.join('' + self._config.data.root, 'train_pos_features.pkl'), 'wb') as f:
            pickle_dump(train_pos_f, f)
        with open(os.path.join('' + self._config.data.root, 'train_pos_len.pkl'), 'wb') as f:
            pickle_dump(train_pos_l, f)
        with open(os.path.join('' + self._config.data.root, 'train_neg_features.pkl'), 'wb') as f:
            pickle_dump(train_neg_f, f)
        with open(os.path.join('' + self._config.data.root, 'train_neg_len.pkl'), 'wb') as f:
            pickle_dump(train_neg_l, f)
        print('finish')

        dev_f, dev_l = self.process_dev_test(self.dev_data)
        np.save(os.path.join('' + self._config.data.root, 'dev_features.npy'), dev_f, allow_pickle=True)
        np.save(os.path.join('' + self._config.data.root, 'dev_len.npy'), dev_l, allow_pickle=True)

        test_f, test_l = self.process_dev_test(self.test_data)
        np.save(os.path.join('' + self._config.data.root, 'test_features.npy'), test_f, allow_pickle=True)
        np.save(os.path.join('' + self._config.data.root, 'test_len.npy'), test_l, allow_pickle=True)

    def find_pos(self, sent_list, word_list):
        '''
        return position list
        '''
        l = len(word_list)
        for i in range(len(sent_list)):
            flag = True
            j = 0
            while j < l:
                if word_list[j] != sent_list[i + j]:
                    flag = False
                    break
                j += 1
            if flag:
                return range(i, i + l)
        return []

    def process_dev_test(self, dataset):
        features = []
        sen_len = []
        for i, data in enumerate(dataset):
            sent_text = data['sentText']
            sent_words, sent_ids, pos_ids, sent_chars, cur_len = self.process_sentence(sent_text)
            entities = data['entityMentions']
            raw_triples_ = data['relationMentions']
            # 去重
            triples_list = []
            for t in raw_triples_:
                triples_list.append((t['em1Text'], t['em2Text'], t['label']))
            triples_ = list(set(triples_list))
            triples_.sort(key=triples_list.index)

            triples = []
            for triple in triples_:
                head, tail, relation = triple
                try:
                    if triple[2] != 'None':
                        head_words = nltk.word_tokenize(head + ',')[:-1]
                        head_pos = self.find_pos(sent_words, head_words)
                        tail_words = nltk.word_tokenize(tail + ',')[:-1]
                        tail_pos = self.find_pos(sent_words, tail_words)
                        h_chunk = ('H', head_pos[0], head_pos[-1] + 1)
                        t_chunk = ('T', tail_pos[0], tail_pos[-1] + 1)
                        triples.append((h_chunk, t_chunk, self.rel2id[relation]))
                except:
                    continue

            features.append([sent_ids, pos_ids, sent_chars, triples])
            sen_len.append(cur_len)
            if (i + 1) * 1.0 % 10000 == 0:
                print('finish %f, %d/%d' % ((i + 1.0) / len(dataset), (i + 1), len(dataset)))
        return np.array(features), np.array(sen_len)

    def process_train(self, dataset):
        positive_features = []
        positive_lens = []
        negative_features = []
        negative_lens = []
        c = 0
        for i, data in enumerate(dataset):
            positive_feature = []
            positive_len = []
            negative_feature = []
            negative_len = []
            sent_text = data['sentText']
            # sent_chars : (max_len, max_word_len)
            sent_words, sent_ids, pos_ids, sent_chars, cur_len = self.process_sentence(sent_text)
            entities_ = data['entityMentions']
            entities = []
            for e_ in entities_:
                entities.append(e_['text'])

            raw_triples_ = data['relationMentions']
            # 去重
            triples_list = []
            for t in raw_triples_:
                triples_list.append((t['em1Text'], t['em2Text'], t['label']))
            triples_ = list(set(triples_list))
            triples_.sort(key=triples_list.index)

            triples = []
            cur_relations_list = []
            cur_relations_list.append(0)
            for triple in triples_:
                cur_relations_list.append(self.rel2id[triple[2]])
                head, tail, relation = triple
                try:
                    if triple[2] != 'None':
                        head_words = nltk.word_tokenize(head + ',')[:-1]
                        head_pos = self.find_pos(sent_words, head_words)
                        tail_words = nltk.word_tokenize(tail + ',')[:-1]
                        tail_pos = self.find_pos(sent_words, tail_words)
                        h_chunk = ('H', head_pos[0], head_pos[-1] + 1)
                        t_chunk = ('T', tail_pos[0], tail_pos[-1] + 1)
                        triples.append((h_chunk, t_chunk, self.rel2id[relation]))
                except:
                    continue

            cur_relations = list(set(cur_relations_list))
            cur_relations.sort(key=cur_relations_list.index)

            if len(cur_relations) == 1 and cur_relations[0] == 0:
                continue
            c += 1
            none_label = ['O'] * cur_len + ['X'] * (self._config.data.max_len - cur_len)
            all_labels = {}  # ['O'] * self.max_len

            for triple in triples_:
                head, tail, relation = triple
                rel_id = self.rel2id[relation]
                # cur_label = none_label.copy()
                cur_label = all_labels.get(rel_id, none_label.copy())
                if triple[2] != 'None':
                    # label head
                    head_words = nltk.word_tokenize(head + ',')[:-1]
                    head_pos = self.find_pos(sent_words, head_words)
                    try:
                        if len(head_pos) == 1:
                            cur_label[head_pos[0]] = 'S-H'
                        elif len(head_pos) >= 2:
                            cur_label[head_pos[0]] = 'B-H'
                            cur_label[head_pos[-1]] = 'E-H'
                            for ii in range(1, len(head_pos) - 1):
                                cur_label[head_pos[ii]] = 'I-H'
                    except:
                        continue

                    # label tail
                    tail_words = nltk.word_tokenize(tail + ',')[:-1]
                    tail_pos = self.find_pos(sent_words, tail_words)
                    try:
                        # not overlap enntity
                        if len(tail_pos) == 1:
                            cur_label[tail_pos[0]] = 'S-T'
                        elif len(tail_pos) >= 2:
                            cur_label[tail_pos[0]] = 'B-T'
                            cur_label[tail_pos[-1]] = 'E-T'
                            for ii in range(1, len(tail_pos) - 1):
                                cur_label[tail_pos[ii]] = 'I-T'

                    except:
                        continue
                    all_labels[rel_id] = cur_label
            for ii in all_labels.keys():
                cur_label_ids = [self.label2id[e] for e in all_labels[ii]]
                positive_feature.append([sent_ids, ii, cur_label_ids, pos_ids, sent_chars])
                # positive_triple.append()
                positive_len.append(cur_len)

            none_label_ids = [self.label2id[e] for e in none_label]
            for r_id in range(self._config.data.rel_num):
                if r_id not in cur_relations:
                    negative_feature.append([sent_ids, r_id, none_label_ids, pos_ids, sent_chars])
                    negative_len.append(cur_len)
            if (i + 1) * 1.0 % 10000 == 0:
                print('finish %f, %d/%d' % ((i + 1.0) / len(dataset), (i + 1), len(dataset)))
            positive_features.append(positive_feature)
            positive_lens.append(positive_len)
            negative_features.append(negative_feature)
            negative_lens.append(negative_len)
        print(c)

        return positive_features, positive_lens, negative_features, negative_lens

    def process_sentence(self, sent_text):
        sent_words = nltk.word_tokenize(sent_text)
        sen_len = min(len(sent_words), self._config.data.max_len)
        sent_pos = nltk.pos_tag(sent_words)
        sent_pos_ids = [self.pos2id.get(pos[1], 1) for pos in sent_pos][:sen_len]
        sent_ids = [self.word2id.get(w, 1) for w in sent_words][:sen_len]

        sent_chars = []
        for w in sent_words[:sen_len]:
            tokens = [self.char2id.get(token, 1) for token in list(w)]
            word_len = min(len(tokens), self._config.data.max_word_len)
            for _ in range(self._config.data.max_word_len - word_len):
                tokens.append(0)
            sent_chars.append(tokens[: self._config.data.max_word_len])

        for _ in range(sen_len, self._config.data.max_len):
            sent_ids.append(0)
            sent_pos_ids.append(0)
            sent_chars.append([0] * self._config.data.max_word_len)
        return sent_words[:sen_len], sent_ids, sent_pos_ids, sent_chars, sen_len


class Data(Dataset):

    def __init__(self, root, prefix):
        self.prefix = prefix
        self.features = np.load(os.path.join(root, prefix + '_features.npy'), allow_pickle=True)
        self.sen_len = np.load(os.path.join(root, prefix + '_len.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sen_len = self.sen_len[idx]
        feature = self.features[idx]
        return feature[0], feature[1], feature[2], feature[3], sen_len


class Train_data(Dataset):

    def __init__(self, config):
        self._config = config
        with open(os.path.join(self._config.data.root, 'train_pos_features.pkl'), 'rb') as f:
            self.features_pos = pickle_load(f)

        with open(os.path.join(self._config.data.root, 'train_pos_len.pkl'), 'rb') as f:
            self.len_pos = pickle_load(f)

        with open(os.path.join(self._config.data.root, 'train_neg_features.pkl'), 'rb') as f:
            self.features_neg = pickle_load(f)

        with open(os.path.join(self._config.data.root, 'train_neg_len.pkl'), 'rb') as f:
            self.len_neg = pickle_load(f)

    def __len__(self):
        assert len(self.features_neg) == len(self.features_pos), 'length invalid'
        assert len(self.len_pos) == len(self.len_neg), 'length invalid'
        assert len(self.features_pos) == len(self.len_pos), 'length invalid'
        return len(self.len_pos)

    def __getitem__(self, idx):
        pos_f = self.features_pos[idx]
        pos_l = self.len_pos[idx]
        neg_f = self.features_neg[idx]
        neg_l = self.len_neg[idx]
        neg_zip = zip(neg_f, neg_l)
        # neg_num = int(len(neg_f)*self.opt.neg_rate)
        neg_num = self._config.data.neg_num
        if neg_num != 0:
            neg_sam = random.sample(list(neg_zip), neg_num)
            neg_fs, neg_ls = zip(*neg_sam)
            example_f = pos_f + list(neg_fs)
            example_l = pos_l + list(neg_ls)
        else:
            example_f = pos_f
            example_l = pos_l
        sents, rels, labels, poses, chars = zip(*example_f)
        return sents, rels, labels, poses, chars, example_l


def dev_test_collate(features):
    sent = []
    triples = []
    poses = []
    chars = []
    sen_len = []
    for feature in features:
        sent.append(torch.tensor(feature[0]))
        poses.append(torch.tensor(feature[1]))
        chars.append(torch.tensor(feature[2]))
        triples.append(feature[3])
        sen_len.append(feature[4])
    sent = torch.stack(sent)
    poses = torch.stack(poses)
    chars = torch.stack(chars)
    sen_len = torch.tensor(sen_len)
    return sent, triples, poses, chars, sen_len


def train_collate(features):
    sent = []
    rel = []
    label = []
    pos = []
    chars = []
    sen_len = []
    for feature in features:
        sent.append(torch.tensor(feature[0]))
        rel.append(torch.tensor(feature[1]))
        label.append(torch.tensor(feature[2]))
        pos.append(torch.tensor(feature[3]))
        chars.append(torch.tensor(feature[4]))
        sen_len.append(torch.tensor(feature[5]))
    sent = torch.cat(sent, 0)
    rel = torch.cat(rel, 0)
    label = torch.cat(label, 0)
    pos = torch.cat(pos, 0)
    chars = torch.cat(chars, 0)
    sen_len = torch.cat(sen_len, 0)
    return sent, rel, label, pos, chars, sen_len


class Loader():
    def __init__(self, config):
        self._config = config

        self.train_data = Train_data(self._config)
        self.train_len = self.train_data.__len__()

        self.dev_data = Data(self._config.data.root, 'dev')
        self.dev_len = self.dev_data.__len__()

        self.test_data = Data(self._config.data.root, 'test')
        self.test_len = self.test_data.__len__()
        self.loader = {}
        self.reset('train')
        self.reset('dev')
        self.reset('test')

    def reset(self, prefix):
        if prefix == 'train':
            self.loader[prefix] = iter(DataLoader(self.train_data, batch_size=1, collate_fn=train_collate,
                                                  shuffle=True))

        if prefix == 'dev':
            self.loader[prefix] = iter(
                DataLoader(self.dev_data, batch_size=1, collate_fn=dev_test_collate, shuffle=False))

        if prefix == 'test':
            self.loader[prefix] = iter(
                DataLoader(self.test_data, batch_size=1, collate_fn=dev_test_collate, shuffle=False))

    def get_batch_train(self, batch_size):
        wrapped = False
        sents = []
        rels = []
        labels = []
        poses = []
        all_chars = []
        sen_lens = []
        for i in range(batch_size):
            try:
                sent, rel, label, pos, chars, sen_len = self.loader['train'].next()
            except:
                self.reset('train')
                sent, rel, label, pos, chars, sen_len = self.loader['train'].next()
                wrapped = True
            sents.append(sent)
            rels.append(rel)
            labels.append(label)
            poses.append(pos)
            all_chars.append(chars)
            sen_lens.append(sen_len)
        sents = torch.cat(sents, 0)
        rels = torch.cat(rels, 0)
        labels = torch.cat(labels, 0)
        poses = torch.cat(poses, 0)
        all_chars = torch.cat(all_chars, 0)
        sen_lens = torch.cat(sen_lens, 0)
        return sents, rels, labels, poses, all_chars, sen_lens, wrapped

    def get_batch_dev_test(self, batch_size, prefix):
        wrapped = False
        sents = []
        gts = []
        poses = []
        chars = []
        sen_lens = []
        for i in range(batch_size):
            try:
                sent, triple, pos, char, sen_len = self.loader[prefix].next()
            except:
                self.reset(prefix)
                sent, triple, pos, char, sen_len = self.loader[prefix].next()
                wrapped = True
            sents.append(sent[0])
            gts.append(triple[0])
            poses.append(pos[0])
            chars.append(char[0])
            sen_lens.append(sen_len[0])
        sents = torch.stack(sents)
        poses = torch.stack(poses)
        chars = torch.stack(chars)
        sen_lens = torch.stack(sen_lens)
        return sents, gts, poses, chars, sen_lens, wrapped
