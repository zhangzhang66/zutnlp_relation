#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : relation_evaluator
# @Author   : ZhiYi Zhang
# @Time     : 2020/10/22 20:18

import torch
import numpy as np
import numpy
from sequence.relation.relation_model import device


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    content = tag_name.split('-')
    tag_class = content[0]
    # tag_type = content[1]
    if len(content) == 1:
        return tag_class
    ht = content[-1]
    return tag_class, ht


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position
    Args:
        seq: np.array[4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    # print(seq)
    # print(tags)
    default1 = tags['O']
    default2 = tags['X']
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if (tok == default1 or tok == default2) and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default1 and tok != default2:
            res = get_chunk_type(tok, idx_to_tag)
            if len(res) == 1:
                continue
            tok_chunk_class, ht = get_chunk_type(tok, idx_to_tag)
            tok_chunk_type = ht
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def attn_mapping(attn_scores, gts):
    gt_rel = []
    for gt in gts:
        gt_rel.append(gt[2] - 1)
    return attn_scores[gt_rel]


def tag_mapping(predict_tags, cur_relation, label2id):
    '''
    parameters
        predict_tags : np.array, shape: (rel_number, max_sen_len)
        cur_relation : list of relation id
    '''
    assert predict_tags.shape[0] == len(cur_relation)

    predict_triples = []
    for i in range(predict_tags.shape[0]):
        heads = []
        tails = []
        pred_chunks = get_chunks(predict_tags[i], label2id)
        for ch in pred_chunks:
            if ch[0].split('-')[-1] == 'H':
                heads.append(ch)
            elif ch[0].split('-')[-1] == 'T':
                tails.append(ch)
        # if heads.qsize() == tails.qsize():
        '''
        # TODO：当前策略：同等匹配，若头尾数量不符则舍弃多出来的部分
        '''
        if len(heads) != 0 and len(tails) != 0:
            if len(heads) < len(tails):
                heads += [heads[-1]] * (len(tails) - len(heads))
            if len(heads) > len(tails):
                tails += [tails[-1]] * (len(heads) - len(tails))

        for h_t in zip(heads, tails):
            # print(h_t)
            ht = list(h_t) + [cur_relation[i]]
            ht = tuple(ht)
            predict_triples.append(ht)
    return predict_triples


def eval(correct_preds, total_preds, total_gt):
    '''
    Evaluation
    :parameter
    :parameter
    :return: P,R,F1
    '''
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_gt if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    return p, r, f1

correct_preds = 0.
total_preds = 0.
total_gt = 0.
def evaluate(config, model,Loader,  label2id, batch_size, rel_num):
    model.eval()
    loader = Loader
    n = 0
    predictions = []
    final_attn = []
    targets = []
    metrics = {}
    correct_preds = 0.
    total_preds = 0.
    total_gt = 0.

    val_num = loader.dev_len

    for i in range(69):
        with torch.no_grad():
            sents, gts, poses, chars, sen_lens, wrapped = Loader.get_batch_dev_test(batch_size)
            sents = sents.to(device)
            sen_lens = sen_lens.cpu()
            sen_lens = sen_lens.numpy().tolist()
            mask = torch.zeros(sents.size()).cpu()
            mask = mask.numpy().tolist()
            poses = poses.to(device)
            chars = chars.to(device)
            n = n + batch_size
            for j in range(len(sen_lens)):
                sen_lens[j] = int(sen_lens[j])
            for i in range(sents.size(0)):
                for j in range(sen_lens[i]):
                    mask[i][j] = 1
            mask = torch.Tensor(mask).to(device)
            sen_lens = torch.Tensor(sen_lens).to(device)

            sents = sents.repeat([1, rel_num - 1]).view(batch_size * (rel_num - 1), -1).to(device)
            poses = poses.repeat([1, rel_num - 1]).view(batch_size * (rel_num - 1), -1).to(device)
            chars = chars.repeat([1, rel_num - 1, 1]).view(batch_size * (rel_num - 1), config.data.max_len, -1).to(device)
            mask = mask.repeat([1, rel_num - 1]).view(batch_size * (rel_num - 1), -1).to(device)
            sen_lens = sen_lens.unsqueeze(1).repeat([1, rel_num - 1]).view(batch_size * (rel_num - 1)).to(device)
            rel = torch.arange(1, rel_num).repeat(batch_size).to(device)
            sents = sents.type(torch.IntTensor).to(device)
            # poses = poses.type(torch.IntTensor).to(device)
            sen_lens = sen_lens.type(torch.IntTensor).to(device)
            if not config.model.use_char:
                chars = None
            if not config.model.use_pos:
                poses = None
            predict, attention_score = model(sents, sen_lens, rel, mask, poses,
                                             chars)  # (batch * rel_num-1) * max_sen_len * label_num
            predict = torch.softmax(predict, -1)

            for i in range(predict.size(0)):
                predict[i][:sen_lens[i], -1] = -1e9
                predict[i][sen_lens[i]:, -1] = 1e9
            decode_tags = numpy.array(predict.max(-1)[1].data.cpu())
            current_relation = [k for k in range(1, rel_num)]

            for i in range(batch_size):
                triple = tag_mapping(decode_tags[i * (rel_num - 1):(i + 1) * (rel_num - 1)], current_relation, label2id)
                # att = attn_mapping(attention_score[i * (rel_num - 1):(i + 1) * (rel_num - 1)], gts[i])
                target = gts[i]
                predictions.append(triple)
                targets.append(target)

                if n - batch_size + i + 1 <= val_num:
                    '''
                    print('Sentence %d:' % (n - batch_size + i + 1))
                    print('predict:')
                    print(triple)
                    print('target:')
                    print(target)
                    '''
                    correct_preds += len(set(triple) & set(target))
                    total_preds += len(set(triple))
                    total_gt += len(set(target))

            if n >= val_num:
                for i in range(n - val_num):
                    predictions.pop()
                    targets.pop()
                p, r, f1, = eval(correct_preds, total_preds, total_gt)
                metrics['P'] = p
                metrics['R'] = r
                metrics['F1'] = f1
                print('test precision {}, recall {}, f1 {}'.format(p, r, f1))
                break
    model.train()
    return predictions, targets, None, metrics
