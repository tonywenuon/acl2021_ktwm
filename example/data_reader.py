
import sys, os
project_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
if project_path not in sys.path:
    sys.path.append(project_path)

import argparse
import numpy as np
from copy import deepcopy
from typing import Callable, Optional, Sequence, Iterable
from run_script.args_parser import kwfm_add_arguments
from commonly_used_code import helper_fn
from commonly_used_code.config import Config

class Batch:
    def __init__(self):
        self.init_all()

    def init_all(self):
        self.init_src_tar()
        self.init_fact()

    def init_src_tar(self):
        # normal ids
        self.sample_ids = []
        self.src_ids = []
        self.tar_ids = []
        self.tar_loss_ids = []


    def init_fact(self):
        self.text_facts = []
        self.fact_ids = []
        self.fact_switches = []

    def np_format(self):
        self.src_ids = np.asarray(self.src_ids)
        self.sample_ids = np.asarray(self.sample_ids)
        self.tar_ids = np.asarray(self.tar_ids)
        self.tar_loss_ids = np.asarray(self.tar_loss_ids)
        self.tar_loss_ids = np.reshape(self.tar_loss_ids, (self.tar_loss_ids.shape[0], self.tar_loss_ids.shape[1], 1))
        self.fact_ids = np.asarray(self.fact_ids)
        self.fact_switches = np.asarray(self.fact_switches)

class DataSet:
    def __init__(self, args):
        self.args = args
        self.config = Config(args.data_set)
        self.__set_file_path()

        # get global token and ids 
        self.src_token_ids, self.src_id_tokens, self.src_vocab_size = self.__read_global_ids(self.src_global_token_path)
        self.tar_token_ids, self.tar_id_tokens, self.tar_vocab_size = self.__read_global_ids(self.tar_global_token_path)
        self.train_sample_num = 0
        self.valid_sample_num = 0
        self.test_sample_num = 0

        self.max_src_len = args.src_seq_length
        self.max_tar_len = args.tar_seq_length
        if not args.fact_seq_length:
            self.max_fact_len = args.src_seq_length
        else:
            print('This is fact length')
            self.max_fact_len = args.fact_seq_length
        self.src_len = self.max_src_len - self.config.src_reserved_pos
        self.tar_len = self.max_tar_len - self.config.tar_reserved_pos

        self.pad_fact_seqs = helper_fn.pad_with_pad([self.pad_id], self.max_fact_len, self.pad_id)

        self.__get_sample_numbers()

    def __set_file_path(self):
        self.train_set_path = self.config.train_qa_path
        self.valid_set_path = self.config.valid_qa_path
        self.test_set_path = self.config.test_qa_path

        self.train_sent_fact_path = self.config.train_oracle_sent_fact_path
        self.valid_sent_fact_path = self.config.valid_oracle_sent_fact_path
        self.test_sent_fact_path = self.config.test_sent_fact_path

        self.src_global_token_path = self.config.global_src_token_path
        self.tar_global_token_path = self.config.global_tar_token_path


    def __get_sample_numbers(self):
        print('Getting total samples numbers...')
        if os.path.exists(self.train_set_path):
            with open(self.train_set_path) as f:
                for line in f:
                    self.train_sample_num += 1
        if os.path.exists(self.valid_set_path):
            with open(self.valid_set_path) as f:
                for line in f:
                    self.valid_sample_num += 1
        with open(self.test_set_path) as f:
            for line in f:
                self.test_sample_num += 1
                
    def _line2ids(self, line, max_len):
        seq = []
        for token in line.strip().split(' '):
            _id = self.src_token_ids.get(token, self.unk_id)
            seq.append(_id)

        seq = seq[:max_len]
        return seq

    def _deal_qa_line(self, index, line):
        elems = line.strip().split('\t')

        text = elems[1].strip()
        seq = self._line2ids(text, self.src_len)
        src_seq = deepcopy(seq)
        src = helper_fn.pad_with_start_end(seq, self.max_src_len, self.start_id, self.end_id, self.pad_id)

        # used for multi_task. If there is no fact, use src as the answer
        seq = self._line2ids(text, self.tar_len)
        src_tar = helper_fn.pad_with_start(seq, self.max_tar_len, self.start_id, self.pad_id)
        src_tar_loss = helper_fn.pad_with_end(seq, self.max_tar_len, self.end_id, self.pad_id)

        text = elems[2].strip()
        seq = self._line2ids(text, self.tar_len)
        tar_seq = deepcopy(seq)
        tar = helper_fn.pad_with_start(seq, self.max_tar_len, self.start_id, self.pad_id)

        tar_loss = helper_fn.pad_with_end(seq, self.max_tar_len, self.end_id, self.pad_id)

        return src, tar, tar_loss, src_tar, src_tar_loss, tar_seq

    def _deal_fact_line(self, index, line, tar_seq):
        line = line.strip()
        seqs = []
        cur_facts = []
        cur_fact_ids = []
        cur_switches = []
        fact_tar = None
        fact_tar_loss = None
        elems = line.split('\t')
        # if there is no fact, add pad sequence
        if len(elems) <=1 or elems[1] == self.config.NO_FACT or elems[1] == self.config.NO_CONTEXT:
            cur_facts.append(self.pad_fact_seqs)
            cur_fact_ids.append(self.pad_fact_seqs)
            cur_switches.append([0] * self.max_fact_len)
        else:
            for index, text in enumerate(elems[1:]):
                cur_facts.append(text)
                seq = self._line2ids(text, self.max_fact_len)
                seqs.append(seq)
                switch = []
                for _id in seq:
                    if _id in tar_seq:
                        switch.append(1)
                    else:
                        switch.append(0)
                switch = switch + [0] * (self.max_fact_len - len(switch))
                cur_switches.append(switch)

                new_seq = helper_fn.pad_with_pad(seq, self.max_fact_len, self.pad_id)
                cur_fact_ids.append(new_seq)

        # pad fact number
        seqs = seqs[:self.args.fact_number]
        cur_facts = cur_facts[:self.args.fact_number]
        cur_fact_ids = cur_fact_ids[:self.args.fact_number]
        cur_switches = cur_switches[:self.args.fact_number]

        cur_facts = cur_facts + [self.pad_fact_seqs] * (self.args.fact_number - len(cur_facts))
        cur_fact_ids = cur_fact_ids + [self.pad_fact_seqs] * (self.args.fact_number - len(cur_fact_ids))
        cur_switches = cur_switches + [[0] * self.max_fact_len] * (self.args.fact_number - len(cur_switches))

        return cur_fact_ids, fact_tar, fact_tar_loss, cur_switches, cur_facts


    def _fit_model(self, _batch, file_type, model_type, is_last_batch=False):
        '''
        Please carefully choose the output type to fit with your model's inputs
        '''

        if file_type == 'train' or file_type == 'valid':
            is_train = np.ones((_batch.src_ids.shape[0], 1))
        else:
            is_train = np.zeros((_batch.src_ids.shape[0], 1))

        if model_type == 'ktwm':
            if file_type == 'test':
                return ([_batch.src_ids, _batch.tar_ids, _batch.tar_loss_ids, _batch.fact_ids, _batch.fact_switches, is_train, _batch.text_facts], None)
            return     ([_batch.src_ids, _batch.tar_ids, _batch.tar_loss_ids, _batch.fact_ids, _batch.fact_switches, is_train], None)

        else:
            raise ValueError('The input model type: %s is not available. ' \
                'Please chech the file: data_reader.py line: _fit_model' % model_type)

    # This is a data generator, which is suitable for large-scale data set
    def data_generator(self, file_type, model_type):
        '''
        :param file_type: This is supposed to be: train, valid, or test
        :param model_type: This is supposed to be different models' name
        '''
        print('This is in data generator...')
        assert(self.max_src_len > 0)
        assert(self.max_tar_len > 0)
        assert(self.max_fact_len > 0)
        assert file_type == 'train' or file_type == 'valid' or file_type == 'test'
    
        if file_type == 'train':
            qa_path = self.train_set_path
            fact_path = self.train_sent_fact_path

        elif file_type == 'valid':
            qa_path = self.valid_set_path
            fact_path = self.valid_sent_fact_path
        elif file_type == 'test':
            qa_path = self.test_set_path
            fact_path = self.test_sent_fact_path
        batch = Batch()


        def _read_qa_fact(batch):
            while True:
                f_qa = open(qa_path)
                f_fact = open(fact_path)
                print(qa_path)
                print(fact_path)
                for index, (qa_line, fact_line) in enumerate(zip(f_qa, f_fact)):
                    qa_id = qa_line.strip().split('\t')[0]
                    fact_id = fact_line.strip().split('\t')[0]
                    assert (qa_id == fact_id)

                    src, tar, tar_loss, src_tar, src_tar_loss, tar_seq = \
                        self._deal_qa_line(index, qa_line )
                    facts, fact_tar, fact_tar_loss, fact_switches, text_facts = \
                        self._deal_fact_line(index, fact_line, tar_seq)

                    batch.sample_ids.append(qa_id)
                    batch.src_ids.append(src)
                    batch.tar_ids.append(tar)
                    batch.tar_loss_ids.append(tar_loss)
                    batch.text_facts.append(text_facts)
                    batch.fact_ids.append(facts)
                    batch.fact_switches.append(fact_switches)

                    if ((index + 1) % self.args.batch_size == 0):
                        ret = deepcopy(batch)
                        ret.np_format()
                        batch = Batch()
                        yield self._fit_model(ret, file_type, model_type)

                if (len(batch.src_ids) != 0):
                    ret = deepcopy(batch)
                    ret.np_format()
                    batch = Batch()
                    yield self._fit_model(ret, file_type, model_type, is_last_batch=True)

        return _read_qa_fact(batch)

    def __read_global_ids(self, token_path):
        f = open(token_path)
        token_ids = dict()
        id_tokens = dict()
        vocab_size = 0
        for line in f:
            elems = line.strip().split('\t')
            word = elems[0]
            index = int(elems[1])
            token_ids[word] = index
            id_tokens[index] = word 
            vocab_size += 1
            if vocab_size >= self.args.vocab_size:
                break

        self.start_id = token_ids.get(self.config.START_TOKEN, -1)
        self.end_id = token_ids.get(self.config.END_TOKEN, -1)
        self.pad_id = token_ids.get(self.config.PAD_TOKEN, -1)
        self.unk_id = token_ids.get(self.config.UNK_TOKEN, -1)
        assert(self.start_id != -1)
        assert(self.end_id != -1)
        assert(self.pad_id != -1)
        assert(self.unk_id != -1)

        return token_ids, id_tokens, vocab_size

       

