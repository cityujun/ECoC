import torch
from torch import nn
from torch.nn import utils as nn_utils
import torch.nn.functional as F
import math


class RNNEncoder(nn.Module):
    def __init__(self,
                item_embed_size,
                item_size=None, 
                feedback_size=None,
                feedback_embed_size=None,
                hidden_size=128,
                for_critic=False,
                with_feedback_embed=True,
                concat_embed_size=None,
                dropout_rate=0.25,
        ):
        super(RNNEncoder, self).__init__()
        self.embed_dropout = nn.Dropout(dropout_rate)
        self.hidden_size = hidden_size
        self.item_embed_size = item_embed_size
        self.for_critic = for_critic
        if for_critic:
            assert concat_embed_size is not None
            self.concat_embed_size = concat_embed_size
            self.gru = nn.GRU(concat_embed_size, hidden_size, 1, batch_first=True)
        else:
            self.item_size = item_size # 0 for padding
            self.with_feedback_embed = with_feedback_embed

            # embedding layer
            self.item_embedding = nn.Embedding(item_size, item_embed_size, padding_idx=0)
            self.init_item_embedding()
            
            if with_feedback_embed:
                self.feedback_size = feedback_size
                self.feedback_embed_size = feedback_embed_size
                self.feedback_embedding = nn.Embedding(feedback_size, feedback_embed_size)
            
            self.rnn_embed_size = item_embed_size + feedback_embed_size if with_feedback_embed else item_embed_size
            self.gru = nn.GRU(self.rnn_embed_size, hidden_size, 1, batch_first=True)

        '''[to be customized], self.output_size & self.net
        '''

    def get_sess_embed(self, seq_items, seq_feedbacks=None):
        seq_item_embed = self.item_embedding(seq_items) # (batch_size, max_batch_length, item_embed_dim)
        if self.with_feedback_embed and seq_feedbacks is not None:
            seq_feedback_embed = self.feedback_embedding(seq_feedbacks)
            return self.embed_dropout(torch.cat((seq_item_embed, seq_feedback_embed), -1))
        else:
            return self.embed_dropout(seq_item_embed)
    
    def init_item_embedding(self):
        std = 1. / math.sqrt(self.item_embed_size)
        # nn.init.xavier_normal_(self.item_embedding.weight)
        self.item_embedding.weight.data.uniform_(-std, std)
    
    def forward_with_embed(self, sess_embed, sess_length):
        raise NotImplementedError

    def forward(self, seq_items, seq_feedbacks, seq_length, return_embed=False):
        sess_embed = self.get_sess_embed(seq_items, seq_feedbacks)
        sess_dense = self.forward_with_embed(sess_embed, seq_length)
        if return_embed:
            return sess_embed, sess_dense
        return sess_dense


class GRUEncoder(RNNEncoder):
    def __init__(self, linear_hidden_size=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense_size = self.hidden_size
        self.net = nn.Sequential(
            nn.Linear(self.dense_size, linear_hidden_size),
            nn.ReLU(),
            nn.Linear(linear_hidden_size, self.item_embed_size),
        )

    def forward_with_embed(self, sess_embed, sess_length):
        # init hidden state
        hx = torch.zeros(1, sess_embed.size(0), self.hidden_size).to(sess_embed.device)
        # pack sequence
        packed = nn_utils.rnn.pack_padded_sequence(sess_embed, sess_length.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hx = self.gru(packed, hx)
        # seq_unpacked, lens_unpacked = nn_utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        # output is seq_unpacked, shape is (batch_size, max_batch_length, hidden_dim)

        # sess_embedding = self.embedding_dropout(hx[-1]) # batch_size, gru_hidden_size
        return self.net(hx[-1])


class NARMEncoder(RNNEncoder):
    """Neural Attentive Session Based Recommendation Model Class
    """
    def __init__(self, linear_hidden_size=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # attention mechanism
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)

        self.dense_size = self.hidden_size * 2
        self.net = nn.Sequential(
            nn.Linear(self.dense_size, linear_hidden_size),
            nn.ReLU(),
            nn.Linear(linear_hidden_size, self.item_embed_size),
        )
    
    def forward_with_embed(self, sess_embed, seq_length):
        # init hidden state
        hx = torch.zeros(1, sess_embed.size(0), self.hidden_size).to(sess_embed.device)
        # pack sequence
        packed = nn_utils.rnn.pack_padded_sequence(sess_embed, seq_length.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hx = self.gru(packed, hx)
        unpacked_out, unpacked_lens = nn_utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        # output is unpacked_out, shape is (batch_size, max_batch_length, hidden_size)
        c_global = hx[-1] # batch_size, gru_hidden_size

        ## first version of implementation w.r.t. attention
        # q = self.linear_one(unpacked_out) # (batch_size, max_batch_length, hidden_size)
        # k = self.linear_two(hx[-1]).unsqueeze(1).expand_as(q)
        # tmp = torch.sum(q * k, dim=2) / math.sqrt(self.hidden_size) # batch_size, max_batch_length
        # mask = torch.where(seq_items > 0, torch.tensor([1.], device=seq_items.device), torch.tensor([0.], device=seq_items.device))
        # mask_tmp = torch.where(mask > 0, tmp, torch.tensor([-20.], device=seq_items.device))
        # alpha = torch.softmax(mask_tmp, dim=1) * mask
        # # mass = torch.sum(alpha, dim=1).view(-1, 1)
        # # weight = alpha / mass # batch_size, max_batch_length
        # c_local = torch.sum(alpha.unsqueeze(2).expand_as(unpacked_out) * unpacked_out, 1)

        # [BETTER], second implementation of attention
        q1 = self.linear_one(unpacked_out) # (batch_size, max_batch_length, hidden_size)
        q2 = self.linear_two(hx[-1]).unsqueeze(1).expand_as(q1)
        alpha = self.linear_three(torch.sigmoid(q1 + q2)) # (batch_size, max_batch_length, 1)
        # mask = torch.where(seq_items > 0, torch.tensor([1.], device=sess_embed.device), torch.tensor([0.], device=sess_embed.device))
        max_length = seq_length.max().item()
        mask = torch.FloatTensor([[1.] * elem + [0.] * (max_length - elem) for elem in seq_length]).to(sess_embed.device)
        mask = mask.unsqueeze(2).expand_as(unpacked_out)
        c_local = torch.sum(alpha.expand_as(unpacked_out) * unpacked_out * mask, 1)
        c_local = self.embed_dropout(c_local)
        
        return self.net(torch.cat([c_local, c_global], 1))


class STAMPEncoder(RNNEncoder):
    def __init__(self, linear_hidden_size=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # attention mechanism
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        embed_size = self.concat_embed_size if self.for_critic else self.rnn_embed_size
        self.linear_last = nn.Linear(self.rnn_embed_size, self.hidden_size, bias=True)

        self.dense_size = self.hidden_size * 3
        self.net = nn.Sequential(
            nn.Linear(self.dense_size, linear_hidden_size),
            nn.ReLU(),
            nn.Linear(linear_hidden_size, self.item_embed_size),
        )

    def forward_with_embed(self, sess_embed, seq_length):
        # init hidden state
        hx = torch.zeros(1, sess_embed.size(0), self.hidden_size).to(sess_embed.device)
        # pack sequence
        packed = nn_utils.rnn.pack_padded_sequence(sess_embed, seq_length.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hx = self.gru(packed, hx)
        unpacked_out, unpacked_lens = nn_utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        # output is unpacked_out, shape is (batch_size, max_batch_length, hidden_size)
        c_global = hx[-1] # batch_size, gru_hidden_size

        # attention
        q1 = self.linear_one(unpacked_out) # (batch_size, max_batch_length, hidden_size)
        q2 = self.linear_two(hx[-1]).unsqueeze(1).expand_as(q1)
        alpha = self.linear_three(torch.sigmoid(q1 + q2)) # (batch_size, max_batch_length, 1)
        # mask = torch.where(seq_items > 0, torch.tensor([1.], device=seq_items.device), torch.tensor([0.], device=seq_items.device))
        max_length = seq_length.max().item()
        mask = torch.FloatTensor([[1.] * elem + [0.] * (max_length - elem) for elem in seq_length]).to(sess_embed.device)
        mask = mask.unsqueeze(2).expand_as(unpacked_out)
        c_local = torch.sum(alpha.expand_as(unpacked_out) * unpacked_out * mask, 1)
        c_local = self.embed_dropout(c_local)

        seq_last_embed = sess_embed[torch.arange(sess_embed.shape[0]), seq_length-1]
        c_last = self.linear_last(seq_last_embed)
        
        return self.net(torch.cat([c_local, c_global, c_last], 1))
