# -*- coding: utf-8 -*-
import json
import random

import dynamic_yaml
import nltk
import numpy as np
import torch


def read_json(filename):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data

def write_data(f, data):
    out_data = [json.dumps(d) for d in data]
    for d in out_data:
        f.write(d)
        f.write('\n')
    f.close()

def build_word_dict(data, out_path):
    words = set()
    for a_data in data:
        sent_text = a_data['sentText']
        sent_words = nltk.word_tokenize(sent_text)
        words.update(set(sent_words))
    words = list(words)
    words.insert(0, 'UNK')
    words.insert(0, 'BLANK')

    np.save(out_path, words)
    return words

def split(data):
    test_instance_num = 5000
    idx = random.sample(range(len(data)), test_instance_num)
    assert len(idx) == test_instance_num
    idx = set(idx)
    test_data = []
    train_data = []
    for i, a_data in enumerate(data):
        if i in idx:
            test_data.append(a_data)
        else:
            train_data.append(a_data)

    valid_instance_num = 5000
    valid_data = train_data[:valid_instance_num]
    train_data = train_data[valid_instance_num:]
    assert len(valid_data) == valid_instance_num
    assert len(test_data) == test_instance_num
    assert len(test_data) + len(train_data) + len(valid_data) == len(data)
    return test_data, train_data, valid_data

def run_split(origin_train, train, test, valid):
    data = read_json(origin_train)
    print('splitting')
    test_data, train_data, valid_data = split(data)
    print('saving')
    write_data(open(test, 'w'), test_data)
    write_data(open(train, 'w'), train_data)
    write_data(open(valid, 'w'), valid_data)

def build_rel2id(data, rel2id_file):
    relations = set()
    for a_data in data:
        triples = a_data['relationMentions']
        for triple in triples:
            relation = triple['label']
            if relation != 'None':
                relations.add(relation)
    relations = list(relations)
    relations.insert(0, 'None')
    rel2id = { j: i for i, j in enumerate(relations) }
    json.dump(rel2id, open(rel2id_file, 'w'))
    return rel2id

def build_labels(label2id_file):
    '''
    parameter
    :label2id_file: filename of label2id
    '''
    BIES = ['S', 'B', 'I', 'E']
    #types = ['LOCATION', 'PERSON', 'ORGANIZATION']
    HT = ['H','T']

    labels = []
    # 有关系的实体标签
    for b in BIES:
        #labels.append(b + '-' + t + '-' + rel)
        for ht in HT:
            labels.append(b  + '-' + ht)

    # O标签
    labels.append('O')
    labels.append('X')
    print('labels number: ', len(labels))
    label2id = {j : i for i, j in enumerate(labels)}
    json.dump(label2id, open(label2id_file, 'w'))

    return label2id

def build_tags(path):
    tags = ['<PAD>','CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PST','POS','PRP',
            'PRP$','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB',
            '-LRB-','-RRB-',',','.']
    tag2id = { j : i for i,j in enumerate(tags) }
    with open(path, 'w') as f:
        json.dump(tag2id, f, indent=1)
    f.close()

def build_char(path):
    chars = ['<PAD>','UNK', '(','%','r','>','/','.','}','w','k','7','#','v','=','1',
             '9','g','d','s','6','e','x','c','&','~','o','2',')','8','?','[','f','a',
             ']','i','`','t',':','m','<','p','0','3','$','!','{','l','*','n','j','h',
             'y','u',',','z','+','-','4','\'','_','q','b','5','@',';']
    char2id = { j : i for i,j in enumerate(chars) }
    print(len(char2id))
    with open(path, 'w') as f:
        json.dump(char2id, f, indent=1)
    f.close()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_file = 'relation_config.yml'
    with open(config_file, mode='r', encoding='UTF-8') as f:
        config = dynamic_yaml.load(f)
    train = config.data.train_path
    valid = config.data.valid_path
    test = config.data.test_path
    build_char(config.data.char2id_path)
    build_tags(config.data.pos2id_path)

    word_file = config.data.word_file_path
    rel2id_file = config.data.relation2id_path
    label2id_file = config.data.label2id_path
    train_data = read_json(train)
    dev_data = read_json(train)
    build_word_dict(train_data + dev_data, word_file)
    rel2id = build_rel2id(train_data, rel2id_file)
    build_labels(label2id_file)

def save_relation_freq(data, rel2count_filename):
    relation2count = dict()
    for a_data in data:
        triples = a_data['relationMentions']
        for triple in triples:
            r = triple['label']
            count = relation2count.get(r, 0)
            relation2count[r] = count + 1
    json.dump(relation2count, open(rel2count_filename, 'w'))
    return relation2count


