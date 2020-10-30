#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : relation_runner
# @Author   : ZhiYi Zhang
# @Time     : 2020/10/22 20:17
import csv
import json
import time
import random
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from common.runner.common_runner import CommonRunner
from sequence.relation.relation_config import SequenceConfig
from sequence.relation.relation_data import RelationDataLoader
from sequence.relation.relation_dataset import Loader
from sequence.relation.relation_evaluator import evaluate
from sequence.relation.relation_loss import LossWrapper
from sequence.relation.relation_model import Rel_based_labeling

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 2020

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class RelationRunner(CommonRunner):
    def __init__(self, seq_config_file):
        super(RelationRunner, self).__init__(seq_config_file)
        self._valid_log_fields = ['episode', 'P', 'R', 'F']
        self._max_f1 = -1
        pass

    def _build_config(self):
        self._config = SequenceConfig(self._config_file).load_config()
        pass

    def _build_data(self):
        self._dataloader = RelationDataLoader(self._config)
        self._train_dataloader = self._dataloader.load_train()
        self.dev_iter = self._dataloader.load_valid()
        self._valid_dataloader = Loader(self._config).get_batch_dev_test(self._config.learn.eval_batch_size)
        self._test_dataloader = self._dataloader.load_test()
        pass

    def _build_model(self):
        self._model = Rel_based_labeling(self._config).to(self._config.device)
        pass

    def _build_loss(self):
        self._loss = LossWrapper(self._model, self._config)
        pass

    def train(self):
        # switch to train mode
        self._model.train()
        print("training...")
        with SummaryWriter(logdir=self._summary_log_dir, comment='model') as summary_writer, \
                open(self._valid_log_filepath, mode='w') as valid_log_file:
            valid_log_writer = csv.writer(valid_log_file, delimiter=',')
            valid_log_writer.writerow(self._valid_log_fields)
            for episode in range(self._config.learn.episode):
                self._train_epoch(episode, summary_writer)
                self._valid(episode, valid_log_writer, summary_writer)
                # self._display_result(episode)
                self._scheduler.step()
        pass

    def _train_epoch(self, episode, summary_writer):
        epoch_start = time.time()
        self._model.train()
        batch = 0
        for dict_input in tqdm(self._train_dataloader):
            dict_loss = self._loss(dict_input).to(device)
            # Backward and optimize
            self._optimizer.zero_grad()  # clear gradients for this training step
            batch_loss = dict_loss
            batch_loss.backward()  # back-propagation, compute gradients
            self._optimizer.step()  # apply gradients
            self.global_step += 1
            batch += 1
            if self.global_step % self._config.learn.batch_display == 0:
                # for loss_key, loss_value in dict_loss.items():
                #     summary_writer.add_scalar('loss/' + loss_key, loss_value, self.global_step)
                # summary_writer.flush()
                elapsed = time.time() - epoch_start
                print(self._train_fmt.format(
                    episode + 1, self.global_step, batch,
                    self._config.learn.train_batch_size,
                    batch_loss.item(), elapsed
                ))
                epoch_start = time.time()
            pass

    def valid(self):
        model = self._load_checkpoint()
        self._model.eval()
        for dict_input in tqdm(self._valid_dataloader):
            dict_outputs = self._model(dict_input)
            # self._display_output(dict_output)
            dict_outputs['target_sequence'] = dict_input.tag
            # send batch pred and target
            self._evaluator.evaluate(dict_outputs['outputs'], dict_outputs['target_sequence'].T)
        # get the result
        f1 = self._evaluator.get_eval_output()
        pass

    def _load_label(self, input_label2id):
        label2id = json.load(open(input_label2id, 'r'))
        return label2id

    def _valid(self, episode, valid_log_writer, summary_writer):
        print("begin validating epoch {}...".format(episode + 1))
        # switch to evaluate mode
        self._model.eval()
        label2id = self._load_label(self._config.data.label2id_path)
        predictions, targets, _, metrics = evaluate(self._config, self._model, Loader(self._config), label2id,
                                                    self._config.learn.eval_batch_size, self._config.data.rel_num,
                                                    )
        # get the result
        f1 = metrics['F1']
        if self._max_f1 < f1:
            self._max_f1 = f1
            self._save_checkpoint(episode)
            pass
    pass

    def _display_output(self, dict_outputs):
        batch_data = dict_outputs['batch_data']

        word_vocab = batch_data.dataset.fields['text'].vocab
        tag_vocab = batch_data.dataset.fields['tag'].vocab

        batch_input_sequence = dict_outputs['input_sequence'].T
        import numpy as np
        batch_output_sequence = np.asarray(dict_outputs['outputs']).T
        batch_target_sequence = dict_outputs['target_sequence'].T

        result_format = "{}\t{}\t{}\n"
        for input_sequence, output_sequence, target_sequence in zip(
                batch_input_sequence, batch_output_sequence, batch_target_sequence):
            this_result = ""
            for word, tag, target in zip(input_sequence, output_sequence, target_sequence):
                if word != "<pad>":
                    this_result += result_format.format(
                        word_vocab.itos[word], tag_vocab.itos[tag], tag_vocab.itos[target]
                    )
            print(this_result + '\n')
        pass

    def test(self):
        pass

    def _display_result(self, episode):
        pass


if __name__ == '__main__':
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_file = 'relation_config.yml'

    runner = RelationRunner(config_file)
    runner.train()
    pass
