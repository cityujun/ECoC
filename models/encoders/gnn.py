import torch
from torch import nn
import torch.nn.functional as F
import math


class GNN(nn.Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.linear_i = nn.Linear(self.input_size, self.gate_size, bias=True)
        self.linear_h = nn.Linear(self.hidden_size, self.gate_size, bias=True)
        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        # max_session_unique_items == A.shape[1]
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        # gi = F.linear(inputs, self.w_ih, self.b_ih)
        # gh = F.linear(hidden, self.w_hh, self.b_hh)
        gi = self.linear_i(inputs)
        gh = self.linear_h(hidden)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SRGNNEncoder(nn.Module):
    """SR-GNN
    """
    def __init__(self,
                item_embed_size,
                item_size=None,
                for_critic=False,
                linear_hidden_size=256,
                step=1,
                dropout_rate=0.25,
        ):
        super(SRGNNEncoder, self).__init__()
        self.item_embed_size = item_embed_size
        self.embed_dropout = nn.Dropout(dropout_rate)
        self.gnn = GNN(item_embed_size, step=step)
        self.hidden_size = item_embed_size
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)

        if not for_critic:
            # self.n_node = item_size
            self.item_embedding = nn.Embedding(item_size, item_embed_size, padding_idx=0)
        
        self.dense_size = self.hidden_size * 2
        self.net = nn.Sequential(
            nn.Linear(self.dense_size, linear_hidden_size),
            nn.ReLU(),
            nn.Linear(linear_hidden_size, self.item_embed_size),
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        # self.item_embedding.weight.data.uniform_(-std, std)
        for name, param in self.named_parameters():
            param.data.uniform_(-std, std)

    def get_sess_embed(self, items):
        # compute hidden, input shape: (bs, max_unique_session_item)
        return self.item_embedding(items) # (bs, max_unique_session_item, hidden_size)

    def forward_with_embed(self, sess_embed, sess_tuple):
        sess_embed = self.embed_dropout(sess_embed)
        alias_inputs, A, mask = sess_tuple
        
        hidden = self.gnn(A, sess_embed) # same as before
        get = lambda i: hidden[i][alias_inputs[i]] # max_seq_length, hidden_size
        seq_hidden = torch.stack([get(i) for i in torch.arange(alias_inputs.size(0)).long()])

        # attention
        ht = seq_hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size, hidden_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size, 1, hidden_size
        q2 = self.linear_two(seq_hidden)  # batch_size, seq_length, hidden_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2)) # batch_size, seq_length, 1
        a = torch.sum(alpha * seq_hidden * mask.view(mask.shape[0], -1, 1).float(), 1) # batch_size, hidden_size

        return self.net(torch.cat([a, ht], 1))

    def forward(self, alias_inputs, A, items, mask, return_embed=False):
        sess_tuple = (alias_inputs, A, mask)
        sess_embed = self.get_sess_embed(items)
        sess_dense = self.forward_with_embed(sess_embed, sess_tuple)
        if return_embed:
            return sess_embed, sess_dense
        return sess_dense
