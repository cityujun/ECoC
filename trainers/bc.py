import os, time
import sys
import torch
import numpy as np
from tqdm import tqdm
import random
from .base_trainer import BaseTrainer
# from torch.nn import functional as F


class DeterministicBCTrainer(BaseTrainer):
    def __init__(self, lr=1e-3, l2=1e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                                lr=lr, weight_decay=l2)
        
    def train_epoch(self, debug=False, eval_dataloader=None):
        self.policy.train()
        losses = []
        training_steps = len(self.dataloader.dataset) // self.dataloader.batch_size

        for ii, (encoder_inputs, target_tuple, _) in enumerate(self.dataloader):#, total=training_steps):
            seq_items, seq_actions, seq_length = encoder_inputs
            target_item, _ = target_tuple

            seq_items, seq_actions = seq_items.to(self.device), seq_actions.to(self.device)
            encoder_inputs = (seq_items, seq_actions, seq_length)
            
            dense_outputs = self.policy(encoder_inputs)
            logits = self.compute_logits(dense_outputs)
            ranking_loss = self.ce_loss(logits, target_item.to(self.device))

            self.optimizer.zero_grad()
            ranking_loss.backward()
            self.optimizer.step()
            losses.append(ranking_loss.item())

        mean_training_losses = np.around(np.mean(losses, axis=0), 4)
        self.total_train_steps += training_steps

        eval_steps, mean_eval_loss = self.eval_epoch(eval_dataloader)
        return training_steps, eval_steps, mean_training_losses, mean_eval_loss

    def eval_epoch(self, eval_dataloader):
        self.policy.eval()
        eval_losses = []
        eval_steps = len(eval_dataloader.dataset) // eval_dataloader.batch_size

        with torch.no_grad():
            for ii, (encoder_inputs, target_tuple, _) in enumerate(eval_dataloader):
                seq_items, seq_actions, seq_length = encoder_inputs
                target_item, _ = target_tuple
                encoder_inputs = (seq_items.to(self.device), seq_actions.to(self.device), seq_length)
                dense_outputs = self.policy.act(encoder_inputs)
                logits = self.compute_logits(dense_outputs)
                ranking_loss = self.ce_loss(logits, target_item.to(self.device))
                eval_losses.append(ranking_loss.item())

        mean_loss = np.around(np.mean(eval_losses, axis=0), 4)
        return eval_steps, mean_loss

    def print_training_info(self, epoch_id, interval, n_steps, losses):
        print(f'Epoch {epoch_id}, time: {interval}, total steps: {n_steps}, ranking loss: {losses}')
        # print(f'Epoch {epoch_id}, time: {interval}, total steps: {n_steps}, ranking loss: {losses[0]}, mse loss: {losses[1]}')
