import os, sys
import random
import numpy as np
import torch
from copy import deepcopy
from trainers.dataset import SessionDataset
from trainers.data_utils import gnn_with_next_collate_wrapper
from trainers.utils import get_params_number
from models.encoders import *
from models.ecoc import C3ImitGraphPolicy
from trainers.ecoc_gnn import C3ImitGraphTrainer
# from ope.deploy import Env


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
                        batch_size=256,
                        collate_fn=gnn_with_next_collate_wrapper,
                        num_workers=4,
                        shuffle=True)

valid_dataloader = torch.utils.data.DataLoader(
                        valid_dataset,
                        batch_size=256,
                        collate_fn=gnn_with_next_collate_wrapper,
                        num_workers=6,
                        shuffle=False)

test_dataloader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=256,
                        collate_fn=gnn_with_next_collate_wrapper,
                        num_workers=6,
                        shuffle=False)

if encoder_name == 'gnn':
    state_encoder = SRGNNEncoder(item_embed_size=64,
                                 item_size=train_dataset.item_size+1,
                                 step=1)
    critic_encoder_1 = SRGNNEncoder(item_embed_size=64,
                                    for_critic=True,
                                    step=1)
    critic_encoder_2 = SRGNNEncoder(item_embed_size=64,
                                    for_critic=True,
                                    step=1)
else:
    raise NotImplementedError('Backbone not implemented!')

policy = C3ImitGraphPolicy(state_encoder=state_encoder,
                           critic_encoder_1=critic_encoder_1,
                           critic_encoder_2=critic_encoder_2,
                           lmbda=0.1,
                           dist_sample_k=500)

get_params_number(policy)

trainer = C3ImitGraphTrainer(dataloader=train_dataloader,
                        valid_dataloader=valid_dataloader,
                        test_dataloader=test_dataloader,
                        policy=policy,
                        model_path=os.path.join(model_path, f'HIC_SR_GNN_{int(ETA)}_{int(ALPHA)}_{int(BETA)}'),
                        topk=[5, 10, 20],
                        alpha=ALPHA,
                        beta=BETA,
                        eta=ETA,
                    )
    
trainer.train(20, print_flag=8, test_debug=False, save_model=True)
print('\n\n')

# python -m runs.yelp_ecoc_gnn gnn 2. 1. 0. 30
