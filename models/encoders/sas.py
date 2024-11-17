import torch
from torch import nn
from torch.nn import utils as nn_utils
import torch.nn.functional as F
import math


class PointWiseFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.net = nn.Sequential(
            # nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
            nn.Linear(input_size, hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            # nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout_rate)
        )

    def forward(self, inputs):
        # outputs = self.net(inputs.transpose(-1, -2))
        # outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs = self.net(inputs)
        outputs += inputs
        return outputs


class SASEncoder(nn.Module):
    def __init__(self,
                 item_embed_size,
                 item_size=None,
                 feedback_size=None,
                 feedback_embed_size=None,
                 hidden_size=128,
                 for_critic=False,
                 with_feedback_embed=False,
                 max_seq_len=50,
                 num_blocks=1,
                 num_heads=1,
                 ffn_dropout=0.25,
        ):
        super(SASEncoder, self).__init__()
        self.item_embed_size = item_embed_size
        self.hidden_size = hidden_size
        self.embed_dropout = nn.Dropout(0.25)
        self.with_feedback_embed = with_feedback_embed

        ## core structures
        self.attn_layer_norms = nn.ModuleList() # to be Q for self-attention
        self.attn_layers = nn.ModuleList()
        self.forward_layer_norms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(num_blocks):
            attn_layer_norm = nn.LayerNorm(item_embed_size, eps=1e-8)
            self.attn_layer_norms.append(attn_layer_norm)

            attn_layer =  nn.MultiheadAttention(embed_dim=item_embed_size,
                                                num_heads=num_heads,
                                                batch_first=True)
            self.attn_layers.append(attn_layer)

            fwd_layer_norm = torch.nn.LayerNorm(item_embed_size, eps=1e-8)
            self.forward_layer_norms.append(fwd_layer_norm)

            fwd_layer = PointWiseFeedForward(item_embed_size, hidden_size, ffn_dropout)
            self.forward_layers.append(fwd_layer)
        self.last_layer_norm = nn.LayerNorm(item_embed_size, eps=1e-8)

        if not for_critic:
            self.item_size = item_size # 0 for padding
            self.item_embedding = nn.Embedding(item_size, item_embed_size, padding_idx=0)
            self.init_item_embedding()

            if with_feedback_embed:
                self.feedback_size = feedback_size
                self.feedback_embed_size = feedback_embed_size
                self.feedback_embedding = nn.Embedding(feedback_size, feedback_embed_size)
            
            pos_embed_size = item_embed_size - feedback_embed_size if with_feedback_embed  else item_embed_size
            self.pos_embedding = nn.Embedding(max_seq_len, pos_embed_size)
    
    def init_item_embedding(self):
        std = 1. / math.sqrt(self.item_embed_size)
        self.item_embedding.weight.data.uniform_(-std, std)
    
    def modify_padding(self, seq_items, seq_length):
        new_seq_items = torch.zeros_like(seq_items)
        for i in range(seq_items.shape[0]):
            l = seq_length[i]
            new_seq_items[i, -l: ] = seq_items[i, :l]
        return new_seq_items
    
    def get_sess_embed(self, seq_items, seq_length, seq_feedbacks=None):
        seq_items = self.modify_padding(seq_items, seq_length)
        item_embed = self.item_embedding(seq_items)
        pos_ids = torch.arange(seq_items.shape[1]).expand_as(seq_items).to(seq_items.device)
        pos_embed = self.pos_embedding(pos_ids)
        item_embed, pos_embed = self.embed_dropout(item_embed), self.embed_dropout(pos_embed)
        timeline_mask = torch.BoolTensor(seq_items.cpu() != 0).to(seq_items.device)
        if self.with_feedback_embed and seq_feedbacks is not None:
            seq_feedbacks = self.modify_padding(seq_feedbacks, seq_length)
            feedback_embed = self.feedback_embedding(seq_feedbacks)
            return (item_embed + torch.cat([pos_embed, feedback_embed], dim=2), timeline_mask)
        return (item_embed + pos_embed, timeline_mask)
    
    def forward_with_embed(self, sess_embed_tuple, seq_length=None):
        sess_embed, timeline_mask = sess_embed_tuple
        sess_dense = sess_embed * timeline_mask.unsqueeze(2) # broadcast in last dim

        max_seq_len = sess_dense.shape[1]
        attention_mask = ~torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=sess_dense.device))
        for i in range(len(self.attn_layers)):
            Q = self.attn_layer_norms[i](sess_dense)
            attn_outputs, _ = self.attn_layers[i](Q, sess_embed, sess_embed,
                                                # key_padding_mask=timeline_mask,  #[TODO] loss to be nan
                                                attn_mask=attention_mask)
            sess_dense = Q + attn_outputs

            sess_dense = self.forward_layer_norms[i](sess_dense)
            sess_dense = self.forward_layers[i](sess_dense)
            sess_dense *= timeline_mask.unsqueeze(2)

        return self.last_layer_norm(sess_dense)[:, -1, :]
        
    def forward(self, seq_items, seq_feedbacks, seq_length, return_embed=False):
        sess_embed_tuple = self.get_sess_embed(seq_items, seq_length, seq_feedbacks)
        sess_dense = self.forward_with_embed(sess_embed_tuple, seq_length)
        if return_embed:
            return sess_embed_tuple, sess_dense
        return sess_dense
