import torch
from torch import nn
import torch.nn.functional as F
import math


class GenericCritic(nn.Module):
    def __init__(self, encoder_1, encoder_2, action_size, linear_hidden_size=256):
        super(GenericCritic, self).__init__()
        self.q1_net = encoder_1
        self.q2_net = encoder_2
        assert self.q1_net.item_embed_size == action_size
        self.action_net =  nn.Sequential(
            nn.Linear(action_size, linear_hidden_size),
            nn.ReLU(),
            nn.Linear(linear_hidden_size, action_size),
        )

    def get_q_vec(self, sess_embed, sess_length):
        q_vec_1 = self.q1_net.forward_with_embed(sess_embed, sess_length)
        q_vec_2 = self.q2_net.forward_with_embed(sess_embed, sess_length)
        return torch.min(q_vec_1, q_vec_2)

    def forward(self, sess_embed, sess_length, actions, return_q_vec=False):
        q_vec_1 = self.q1_net.forward_with_embed(sess_embed, sess_length)
        q_vec_2 = self.q2_net.forward_with_embed(sess_embed, sess_length)
        actions = F.normalize(actions, dim=-1, p=2)
        q_val_1 = (q_vec_1 * self.action_net(actions)).sum(dim=-1, keepdim=True)
        q_val_2 = (q_vec_2 * self.action_net(actions)).sum(dim=-1, keepdim=True)
        if return_q_vec:
            return q_val_1, q_val_2, q_vec_1, q_vec_2
        return q_val_1, q_val_2

    def min_q(self, sess_embed, sess_length, actions):
        q_vec = self.get_q_vec(sess_embed, sess_length)
        return self.min_q_with_q_vec(q_vec, actions)
    
    def min_q_with_q_vec(self, q_vec, actions):
        actions = F.normalize(actions, dim=-1, p=2)
        return (q_vec * self.action_net(actions)).sum(dim=-1, keepdim=True)
