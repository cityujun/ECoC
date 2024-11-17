import sys
import os, time
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from common.metrics import get_mrr, get_recall, get_ndcg


class BaseTrainer(object):
    def __init__(self,
                dataloader,
                valid_dataloader,
                test_dataloader,
                policy,
                topk,
                loss_type='CE',
                model_path=None,
        ):
        self.dataloader = dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.policy = policy

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.topk = topk
        self.total_train_steps = 0
        self.total_eval_steps = 0

        self.device = self.policy.device
        self.model_path = model_path

        # self.item_idx = self.dataloader.dataset.item_idx
        # self.item_pop = self.dataloader.dataset.item_popularity
    
    def load(self, policy_model_path):
        self.policy.load_state_dict(torch.load(policy_model_path))
        
    def train_epoch(self, debug, eval_dataloader):
        raise NotImplementedError

    def print_training_info(self, epoch_id, interval, n_steps, losses):
        raise NotImplementedError

    def compute_eval_metrics(self, logits, target_item, topk):    
        _, topk_indices = torch.topk(logits, topk, -1)
        n_recalls = get_recall(topk_indices, target_item)
        n_mrrs = get_mrr(topk_indices, target_item)
        n_ndcg = get_ndcg(topk_indices, target_item)
        n_nums = target_item.size(0)
        return n_recalls, n_mrrs, n_ndcg, n_nums

    def compute_logits(self, dense_outputs):
        item_weights = self.policy.actor.item_embedding.weight
        return torch.matmul(dense_outputs, item_weights.transpose(1, 0))
    
    def train(self, n_epoch, train_debug=False, test_debug=False, save_model=False, print_logger=False, print_flag=5):
        if train_debug or test_debug:
            self.logger = SummaryWriter()
        last_eval_loss, last_policy = None, None
        
        for k in self.topk:
            rec, mrr, ndcg = self.test(self.test_dataloader, k, debug=False)
            print(f'Test of TARGET policy: RECALL@{k} - {rec}, MRR@{k} - {mrr}, NDCG@{k} - {ndcg}')
        
        for ii in range(n_epoch):
            start = time.time()
            training_steps, eval_steps, training_losses, eval_loss = self.train_epoch(train_debug, self.valid_dataloader)
            end = time.time()
            interval = round(end - start, 1)

            self.print_training_info(ii+1, interval, training_steps, training_losses)
            print(f'Eval steps: {eval_steps}, eval loss: {eval_loss}')
            
            if print_logger:
                if ii >= print_flag:
                    for k in self.topk:
                        rec, mrr, ndcg = self.test(self.test_dataloader, k, debug=test_debug)
                        print(f'Test of BC policy: RECALL@{k} - {rec}, MRR@{k} - {mrr}, NDCG@{k} - {ndcg}')
            else:
                cur_eval_loss = eval_loss[0] if isinstance(eval_loss, np.ndarray) else eval_loss
                if last_eval_loss is not None and cur_eval_loss > last_eval_loss:
                    break
                last_eval_loss = cur_eval_loss
            last_policy = self.policy.state_dict()
        
        if save_model and self.model_path is not None and last_policy is not None:
            torch.save(last_policy, self.model_path + '.pth')
        
        if not print_logger:
            for k in self.topk:
                rec, mrr, ndcg = self.test(self.test_dataloader, k, debug=test_debug)
                print(f'Test of BC policy: RECALL@{k} - {rec}, MRR@{k} - {mrr}, NDCG@{k} - {ndcg}')
    
    def test(self, dataloader, topk, debug=False):
        self.policy.eval()
        total_recalls, total_mrrs, total_ndcg, total_nums = 0, 0, 0, 0
        # total_reward = 0
        eval_steps = len(dataloader.dataset) // dataloader.batch_size
        
        for ii, (encoder_inputs, target_tuple, _) in enumerate(dataloader): #, total=eval_steps):
            seq_items, seq_actions, seq_length = encoder_inputs
            target_item, _ = target_tuple
            seq_items, seq_actions = seq_items.to(self.device), seq_actions.to(self.device)
            encoder_inputs = (seq_items, seq_actions, seq_length)
            
            dense_outputs = self.policy.act(encoder_inputs)
            logits = self.compute_logits(dense_outputs).detach()
            
            batch_recalls, batch_mrrs, batch_ndcg, batch_nums = self.compute_eval_metrics(logits, \
                                                                    target_item.to(self.device), topk)
            total_recalls += batch_recalls
            total_mrrs += batch_mrrs
            total_ndcg += batch_ndcg
            total_nums += batch_nums
        
        self.total_eval_steps += eval_steps
        return round(total_recalls / total_nums, 4), round(total_mrrs / total_nums, 4), round(total_ndcg / total_nums, 4)


class BaseRLTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    ## rewrite train for distinct logging print and model saving
    def train(self, n_epoch, train_debug=False, test_debug=False, save_model=False, print_logger=True, print_flag=5):
        if train_debug:
            self.logger = SummaryWriter()
        best_epoch, best_policy = None, None
        best_rec, best_mrr = None, None
        
        for k in self.topk:
            rec, mrr, ndcg = self.test(self.test_dataloader, k, debug=False)
            print(f'Test of TARGET policy: RECALL@{k} - {rec}, MRR@{k} - {mrr}, NDCG@{k} - {ndcg}')
        
        for ii in range(n_epoch):
            start = time.time()
            training_steps, eval_steps, training_losses, eval_loss = self.train_epoch(train_debug, self.valid_dataloader)
            end = time.time()
            interval = round(end - start, 1)

            self.print_training_info(ii+1, interval, training_steps, training_losses)
            print(f'Eval steps: {eval_steps}, eval loss: {eval_loss}')
            
            assert print_logger == True
            if (ii+1) >= print_flag:
                for k in self.topk:
                    rec, mrr, ndcg = self.test(self.test_dataloader, k, debug=test_debug)
                    if test_debug:
                        print(f'Test of DEBUG / TARGET policy: RECALL@{k} - {rec[0]} / {rec[1]}, MRR@{k} - {mrr[0]} / {mrr[1]}, NDCG@{k} - {ndcg[0]} / {ndcg[1]}')
                    else:
                        print(f'Test of TARGET policy: RECALL@{k} - {rec}, MRR@{k} - {mrr}, NDCG@{k} - {ndcg}')
                
                cur_rec = rec[1] if isinstance(rec, tuple) else rec
                cur_mrr = mrr[1] if isinstance(mrr, tuple) else mrr

                if (best_rec is not None and cur_rec >= best_rec) and (best_mrr is not None and cur_mrr >= best_mrr):
                    best_policy = self.policy.state_dict()
                    best_epoch = ii + 1

                best_rec = cur_rec if best_rec is None else max(cur_rec, best_rec)
                best_mrr = cur_mrr if best_mrr is None else max(cur_mrr, best_mrr)
        
        if save_model and self.model_path is not None and best_policy is not None:
            torch.save(best_policy, self.model_path + f'_{best_epoch}.pth')
            print('Model saved. The best n_epoch is: ', best_epoch)
