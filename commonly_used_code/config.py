# !/use/bin/env python

import os
import configparser
from .helper_fn import *

parser = configparser.SafeConfigParser()
config_file_path = '../configuration/config.ini'
parser.read(config_file_path)


class Config:
    def __init__(self, data_set):
        self.data_set = data_set
        # reserve <START> and <END> for source
        self.src_reserved_pos = 2
        # reserve <START> or <END> for target
        self.tar_reserved_pos = 1
        self.NO_FACT = 'no_fact'
        self.NO_CONTEXT = 'no_context'
        self.START_TOKEN = '<start>'
        self.END_TOKEN = '<end>'
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.SPECIAL_TOKENS = [self.START_TOKEN, self.END_TOKEN, self.PAD_TOKEN, self.UNK_TOKEN]

        if data_set == 'wizard':
            self.data_path = parser.get('FilePath', 'wizard_data_path')
        elif data_set == 'full_reduced':
            self.data_path = parser.get('FilePath', 'full_reduced_data_path')

        self.train_path = parser.get('FilePath', 'train_path')
        self.valid_path = parser.get('FilePath', 'valid_path')
        self.test_path = parser.get('FilePath', 'test_path')

        src_global_token_path = parser.get('GenerativeModel', 'src_global_token_dict')
        tar_global_token_path = parser.get('GenerativeModel', 'tar_global_token_dict')
        
        pro_qa_data_path = parser.get('GenerativeModel', 'pro_qa_data')
        oracle_sent_fact_data_path = parser.get('GenerativeModel', 'oracle_sent_fact_data')
        sent_fact_data_path = parser.get('GenerativeModel', 'sent_fact_data')
        

        self.train_path = os.path.join(self.data_path, self.train_path)
        self.valid_path = os.path.join(self.data_path, self.valid_path)
        self.test_path = os.path.join(self.data_path, self.test_path)
        makedirs(self.train_path)
        makedirs(self.valid_path)
        makedirs(self.test_path)

        self.global_src_token_path = os.path.join(self.train_path, src_global_token_path)
        self.global_tar_token_path = os.path.join(self.train_path, tar_global_token_path)

        self.train_qa_path = os.path.join(self.train_path, pro_qa_data_path)
        self.train_oracle_sent_fact_path = os.path.join(self.train_path, oracle_sent_fact_data_path)
        self.train_sent_fact_path = os.path.join(self.train_path, sent_fact_data_path)

        self.valid_qa_path = os.path.join(self.valid_path, pro_qa_data_path)
        self.valid_oracle_sent_fact_path = os.path.join(self.valid_path, oracle_sent_fact_data_path)
        self.valid_sent_fact_path = os.path.join(self.valid_path, sent_fact_data_path)
        
        self.test_qa_path = os.path.join(self.test_path, pro_qa_data_path)
        self.test_sent_fact_path = os.path.join(self.test_path, sent_fact_data_path)



