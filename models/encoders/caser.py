import torch
from torch import nn
from torch.nn import utils as nn_utils
import torch.nn.functional as F
import math
from models.encoders.rnn import RNNEncoder


class CaserEncoder(RNNEncoder):
    def __init__(self,
                 *args,
                 linear_hidden_size=256,
                 n_v=4,
                 n_h=16,
                 L=5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # cnn
        cnn_embed_size = self.concat_embed_size if self.for_critic else self.rnn_embed_size
        self.conv_v = nn.Conv2d(1, n_v, (L, 1))
        self.conv_h = nn.ModuleList([nn.Conv2d(1, n_h, (i+1, cnn_embed_size)) for i in range(L)])
        self.L = L

        self.dense_size = n_v * cnn_embed_size + n_h * L + self.hidden_size
        self.net = nn.Sequential(
            nn.Linear(self.dense_size, linear_hidden_size),
            nn.ReLU(),
            nn.Linear(linear_hidden_size, self.item_embed_size),
        )

    ## rewrite get_sess_embed, output tuple
    def get_sess_embed(self, seq_items, seq_feedbacks, seq_length):
        batch_size = seq_items.size(0)

        # rnn embed
        seq_item_embed = self.item_embedding(seq_items) # (batch_size, max_batch_length, item_embed_dim)
        if self.with_feedback_embed and seq_feedbacks is not None:
            seq_feedback_embed = self.feedback_embedding(seq_feedbacks)
            rnn_embed = self.embed_dropout(torch.cat((seq_item_embed, seq_feedback_embed), -1))
        else:
            rnn_embed = self.embed_dropout(seq_item_embed)

        # cnn embed
        fixed_win_embed = torch.zeros_like(rnn_embed)[:, :self.L]
        for i in range(self.L):
            fixed_win_embed[:, i] = rnn_embed[torch.arange(batch_size), seq_length-(self.L-i)]

        return (rnn_embed, fixed_win_embed)

    def forward_with_embed(self, sess_embed, seq_length):
        rnn_embed, fixed_win_embed = sess_embed
        
        ## cnn_embed -> cnn_dense
        # vertical conv layer
        out_v = self.conv_v(fixed_win_embed.unsqueeze(1)) # (bs, n_v, max_batch_len-(L-1), embedding_size)
        out_v = out_v.view(-1, out_v.size(1) * out_v.size(3)) # (bs, n_v * embedding_size)
        # horizental conv layer
        out_hs = []
        for conv in self.conv_h:
            conv_out = conv(fixed_win_embed.unsqueeze(1)).squeeze(3) # (bs, n_h, max_batch_len-(L-1))
            out_hs.append(F.relu(conv_out).max(dim=-1)[0])
        out_h = torch.cat(out_hs, dim=1)  # (bs, L * n_h)
        
        cnn_dense = torch.cat([out_v, out_h], 1)
        cnn_dense = self.embed_dropout(cnn_dense)

        ## rnn_embed -> rnn_dense
        rnn_embed = self.embed_dropout(rnn_embed) # batch_size, gru_hidden_size
        # init hidden state
        hx = torch.zeros(1, rnn_embed.shape[0], self.hidden_size).to(rnn_embed.device)
        # pack sequence
        packed = nn_utils.rnn.pack_padded_sequence(rnn_embed, seq_length.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, hx = self.gru(packed, hx)
        rnn_dense = hx[-1]

        return self.net(torch.cat((cnn_dense, rnn_dense), 1))

    def forward(self, seq_items, seq_feedbacks, seq_length, return_embed=False):
        sess_embed = self.get_sess_embed(seq_items, seq_feedbacks, seq_length)
        sess_dense = self.forward_with_embed(sess_embed, seq_length)
        if return_embed:
            return sess_embed, sess_dense
        return sess_dense
