import os, sys
import random
import numpy as np
import torch
from copy import deepcopy
from trainers.dataset import SessionDataset
from trainers.data_utils import rnn_collate_wrapper
from trainers.utils import get_params_number
from models.encoders import *
from models.ecoc import C3ImitDuplicatedPolicy
from trainers.ecoc import C3ImitTrainer
# from ope.deploy import Env


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
assert len(sys.argv) == 6
encoder_name = str(sys.argv[1])
ETA = float(sys.argv[2])
ALPHA = float(sys.argv[3])
BETA = float(sys.argv[4])
rd_seed = int(sys.argv[5])

## set random seeds
random.seed(rd_seed)
torch.manual_seed(rd_seed)
torch.cuda.manual_seed(rd_seed)

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

model_path = 'model_path/yelp'
if not os.path.exists(model_path):
    os.mkdir(model_path)

train_dataloader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=512,
                        collate_fn=rnn_collate_wrapper,
                        num_workers=2,
                        shuffle=True,
                        pin_memory=True)

valid_dataloader = torch.utils.data.DataLoader(
                        valid_dataset,
                        batch_size=1024,
                        collate_fn=rnn_collate_wrapper,
                        num_workers=2,
                        shuffle=False,
                        pin_memory=True)

test_dataloader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=1024,
                        collate_fn=rnn_collate_wrapper,
                        num_workers=2,
                        shuffle=False,
                        pin_memory=True)

assert encoder_name in ['gru', 'nextit', 'sas'], "Backbone not implemented!"

if encoder_name == 'gru':
    state_encoder = GRUEncoder(item_size=train_dataset.item_size + 1,
                               item_embed_size=64,
                               feedback_size=train_dataset.action_size,
                               feedback_embed_size=8,
                               hidden_size=128,
                               with_feedback_embed=False,)

    critic_encoder_1 = GRUEncoder(for_critic=True,
                                  concat_embed_size=64,
                                  item_embed_size=64,
                                  hidden_size=128)
    critic_encoder_2 = GRUEncoder(for_critic=True,
                                  concat_embed_size=64,
                                  item_embed_size=64,
                                  hidden_size=128)
    
elif encoder_name == 'nextit':
    state_encoder = NextItEncoder(item_size=train_dataset.item_size+1, 
                                  item_embed_size=64,
                                  num_blocks=2,
                                  dilations=[1, 2, 4, 8],
                                  kernel_size=3)
    
    critic_encoder_1 = NextItEncoder(item_embed_size=64,
                                     for_critic=True,
                                     num_blocks=2,
                                     dilations=[1, 2, 4, 8],
                                     kernel_size=3)
    critic_encoder_2 = NextItEncoder(item_embed_size=64,
                                     for_critic=True,
                                     num_blocks=2,
                                     dilations=[1, 2, 4, 8],
                                     kernel_size=3)
    
elif encoder_name == 'sas':
    state_encoder = SASEncoder(item_size=train_dataset.item_size+1, 
                               item_embed_size=64,
                               with_feedback_embed=False,
                               hidden_size=128, 
                               max_seq_len=50+1, # +1 for s_ in rl policies
                               num_blocks=1,
                               num_heads=2)
    critic_encoder_1 = SASEncoder(for_critic=True, 
                                  item_embed_size=64,
                                  hidden_size=128, 
                                  max_seq_len=50+1, # +1 for s_ in rl policies
                                  num_blocks=1,
                                  num_heads=2)
    critic_encoder_2 = SASEncoder(for_critic=True, 
                                  item_embed_size=64,
                                  hidden_size=128, 
                                  max_seq_len=50+1, # +1 for s_ in rl policies
                                  num_blocks=1,
                                  num_heads=2)
    
policy = C3ImitDuplicatedPolicy(state_encoder=state_encoder,
                                critic_encoder_1=critic_encoder_1,
                                critic_encoder_2=critic_encoder_2,
                                lmbda=0.1,
                                dist_sample_k=500)

get_params_number(policy)
    
trainer = C3ImitTrainer(dataloader=train_dataloader,
                        valid_dataloader=valid_dataloader,
                        test_dataloader=test_dataloader,
                        policy=policy,
                        model_path=os.path.join(model_path, f'HIC_{encoder_name.upper()}_{int(ETA)}_{int(ALPHA)}_{int(BETA)}'),
                        topk=[5, 10, 20],
                        alpha=ALPHA,
                        beta=BETA,
                        eta=ETA,
                    )

epoch_dict = {'gru': 25, 'nextit': 15, 'sas': 25}
flag_dict = {'gru': 10, 'nextit': 5, 'sas': 10}

trainer.train(epoch_dict[encoder_name],
              print_flag=flag_dict[encoder_name],
              test_debug=False,
              save_model=True
            )

print('\n\n')
#  python -m runs.yelp_ecoc gru 2. 1. 0. 30