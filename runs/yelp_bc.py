import os, sys
import numpy as np
import torch
from copy import deepcopy
from trainers.dataset import SessionDataset
from trainers.data_utils import rnn_collate_wrapper
from models import encoders
from models.encoders import SASEncoder, NextItEncoder
from models.base_policy import BasePolicy as BCPolicy
from trainers.bc import DeterministicBCTrainer as BCTrainer
# from ope.deploy import Env

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
assert len(sys.argv) == 2
encoder_name = str(sys.argv[1])

train_file = 'data/yelp/v1_train.csv'
valid_file = 'data/yelp/v1_valid.csv'
test_file = 'data/yelp/v1_test.csv'
train_dataset = SessionDataset(train_file, valid_file)
valid_dataset = deepcopy(train_dataset)
test_dataset = SessionDataset(train_file, test_file)

train_dataset.prepare_data('train', 5, 50)
valid_dataset.prepare_data('valid', 5, 50)
test_dataset.prepare_data('test', 5, 50)
assert sum(np.in1d(train_dataset.test_df.ItemId, train_dataset.train_df.ItemId)) == len(train_dataset.test_df)

train_dataloader = torch.utils.data.DataLoader(
                                train_dataset, 
                                batch_size=512,
                                collate_fn=rnn_collate_wrapper,
                                num_workers=2,
                                multiprocessing_context="fork",
                                shuffle=True)

valid_dataloader = torch.utils.data.DataLoader(
                                valid_dataset, 
                                batch_size=1024,
                                collate_fn=rnn_collate_wrapper,
                                num_workers=2,
                                multiprocessing_context="fork",
                                shuffle=False)

test_dataloader = torch.utils.data.DataLoader(
                                test_dataset, 
                                batch_size=2048,
                                collate_fn=rnn_collate_wrapper,
                                num_workers=2,
                                multiprocessing_context="fork",
                                shuffle=False)

model_path = 'model_path/yelp'
if not os.path.exists(model_path):
    os.mkdir(model_path)

if encoder_name in ['gru', 'narm', 'stamp', 'caser']:
    state_encoder = getattr(encoders, encoder_name.upper() + 'Encoder')(
                                item_embed_size=64,
                                item_size=train_dataset.item_size+1,
                                hidden_size=128,
                                with_feedback_embed=False,
                                # feedback_size=train_dataset.action_size,
                                # feedback_embed_size=8,
    )

elif encoder_name == 'sas':
    # SASRec
    state_encoder = SASEncoder(item_size=train_dataset.item_size+1, 
                               item_embed_size=64,
                               with_feedback_embed=False,
                               hidden_size=128,
                               max_seq_len=50,
                               num_blocks=1,
                               num_heads=2)

elif encoder_name == 'nextit':
    state_encoder = NextItEncoder(item_size=train_dataset.item_size+1,
                                  item_embed_size=64,
                                  num_blocks=2,
                                  dilations=[1, 2, 4, 8],
                                  kernel_size=3)

else:
    raise NotImplementedError('Encoder does not exist!')


dtm_policy = BCPolicy(state_encoder=state_encoder)
dtm_policy.to(dtm_policy.device)

dtm_trainer = BCTrainer(dataloader=train_dataloader,
                        valid_dataloader=valid_dataloader,
                        test_dataloader=test_dataloader,
                        policy=dtm_policy,
                        model_path=os.path.join(model_path, f'BC_{encoder_name.upper()}'),
                        topk=[5, 10, 20])

dtm_trainer.train(20, save_model=True)
