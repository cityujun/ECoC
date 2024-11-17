import os, sys
import numpy as np
import torch
from copy import deepcopy
from trainers.dataset import SessionDataset
from trainers.data_utils import gnn_collate_wrapper
from models.encoders import SRGNNEncoder
from models.base_policy import BasePolicy as BCPolicy
from trainers.bc_gnn import DeterministicBCTrainer as BCTrainer
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

model_path = 'model_path/yelp'
if not os.path.exists(model_path):
    os.mkdir(model_path)

train_dataloader = torch.utils.data.DataLoader(
                                train_dataset, 
                                batch_size=256,
                                collate_fn=gnn_collate_wrapper,
                                num_workers=6,
                                shuffle=True)

valid_dataloader = torch.utils.data.DataLoader(
                                valid_dataset, 
                                batch_size=256,
                                collate_fn=gnn_collate_wrapper,
                                num_workers=8,
                                shuffle=False)

test_dataloader = torch.utils.data.DataLoader(
                                test_dataset, 
                                batch_size=256,
                                collate_fn=gnn_collate_wrapper,
                                num_workers=8,
                                shuffle=False)

if encoder_name == 'srgnn':
    ## vanilla SR-GNN
    state_encoder = SRGNNEncoder(item_embed_size=64,
                                 item_size=train_dataset.item_size+1,
                                 step=1)

    dtm_policy = BCPolicy(state_encoder=state_encoder)
    dtm_policy.to(dtm_policy.device)

    dtm_trainer = BCTrainer(dataloader=train_dataloader,
                            valid_dataloader=valid_dataloader,
                            test_dataloader=test_dataloader,
                            policy=dtm_policy,
                            model_path=os.path.join(model_path, 'BC_SRGNN'),
                            topk=[5, 10, 20])

    dtm_trainer.train(15, save_model=True)

else:
    raise NotImplementedError('Encoder does not exist!')
