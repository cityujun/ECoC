import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset


class SessionDataset(Dataset):
    def __init__(self, train_file, test_file, sep='\t',
                session_key='SessionId', user_key='UserId', item_key='ItemId', 
                action_key='Score', time_key='TimeStr'):
        # Read csv
        self.train_df = pd.read_csv(train_file, sep=sep)
        self.train_df.sort_values([session_key, time_key], inplace=True, ignore_index=True)
        self.test_df = pd.read_csv(test_file, sep=sep)
        self.test_df.sort_values([session_key, time_key], inplace=True, ignore_index=True)
            
        self.session_key = session_key
        self.user_key = user_key
        self.item_key = item_key
        self.time_key = time_key
        self.action_key = action_key
        self.positive_threshold = 1
        self.train_df[action_key] = self.train_df[action_key] - 3
        self.test_df[action_key] = self.test_df[action_key] - 3

        self.add_itemmap()

    def __len__(self):
        return self.all_num

    def __getitem__(self, idx):
        seq_items = [ item for (item, _) in self.all_inputs[idx] ]
        seq_actions = [ action for (_, action) in self.all_inputs[idx] ]
        target, label = self.all_labels[idx]
        terminal = self.all_terminals[idx]
        return (seq_items, seq_actions, target, label, terminal)

    def prepare_data(self, 
                    mode,
                    lower_window=5,
                    upper_window=50,
        ):
        df = self.train_df if mode == 'train' else self.test_df
        assert df.groupby(self.session_key).size().min() > lower_window 
        n_sess = df[self.session_key].nunique()
        self.all_inputs, self.all_labels = [], []
        self.pos_num, self.neg_num = 0, 0
        # sess_len , pos_per_sess = [], []
        self.all_terminals = []

        print('Preparing dataloader ...')
        for (sess_id, grouped) in df.groupby(self.session_key):#, total=n_sess):
            # sess_len.append(len(grouped))

            item_ids = grouped[self.item_key].values
            action_ids = grouped[self.action_key].values
            # pos_per_sess.append(sum(action_ids >= 1))
            item_action_tuples = [(self.item2idx[item_id], action_id) \
                                for (item_id, action_id) in zip(item_ids, action_ids)
                            ]
            
            for end_idx in range(lower_window, len(item_ids)):
                start_idx = max(0, end_idx - upper_window)
                label = item_action_tuples[end_idx][1]
                self.all_inputs.append(item_action_tuples[start_idx: end_idx])
                self.all_labels.append(item_action_tuples[end_idx])
                # self.all_terminals.append(float(random.random()<0.5))
                terminal = 0. if (end_idx + 1) % 5 == 0 else 1.
                self.all_terminals.append(terminal)
                # non click as positive samples
                if label >= self.positive_threshold:  
                    self.pos_num += 1
                else:
                    self.neg_num += 1

        self.all_num = self.pos_num + self.neg_num
        self.print_info(df)
        print(f'positive samples: {self.pos_num}, negative samples: {self.neg_num}')
        # print(f'session number: {len(sess_len)}')
        # print(f'avg. session length: {sum(sess_len)/len(sess_len)}, avg. pos fb: {sum(pos_per_sess) / len(pos_per_sess)}')
    
    def add_itemmap(self):
        self.item_ids = self.train_df[self.item_key].unique()  # type is numpy.ndarray
        self.item2idx = dict()
        self.idx2item = dict()
        for i, item_id in enumerate(self.item_ids):
            self.item2idx[item_id] = i + 1
            self.idx2item[i+1] = item_id

    @property
    def items(self):
        return self.item_ids

    @property
    def item_idx(self):
        return [self.item2idx[item_id] for item_id in self.items]

    @property
    def item_size(self):
        return len(self.item_ids)

    @property
    def action_size(self):
        return self.train_df[self.action_key].nunique()

    @property
    def item_popularity(self):
        item_pop = self.train_df.groupby(self.item_key).size()
        idx_pop = [item_pop[self.idx2item[idx]] for idx in self.item_idx]
        sum_pop = sum(idx_pop)
        idx_pop = [pop / sum_pop for pop in idx_pop]
        return idx_pop

    def print_info(self, df):
        sess_grouped = df.groupby(self.session_key).size()
        item_grouped = df.groupby(self.item_key).size()
        
        print('Dataset info:')
        print(f'Item size: {df[self.item_key].nunique()}, action size: {df[self.action_key].nunique()}')
        print(f'item support range - {item_grouped.min()} / {item_grouped.max()}')
        print(f'session length range - {sess_grouped.min()} / {sess_grouped.max()}')
