import torch
from torch import nn
from torch.nn import utils as nn_utils
import torch.nn.functional as F
import math


class ResidualBlock_b(nn.Module):
    """
    Residual block (b) in the paper
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(ResidualBlock_b, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=(1, kernel_size),
            padding=0,
            dilation=dilation,
        )
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size=(1, kernel_size),
            padding=0,
            dilation=dilation * 2,
        )
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)

        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out = F.relu(self.ln2(out))
        return out + x

    def conv_pad(self, x, dilation):
        """Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
        inputs_pad = inputs_pad.unsqueeze(2)  # [batch_size, embed_size, 1, seq_len]
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        return inputs_pad


class NextItEncoder(nn.Module):
    """Please refer to
    https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/nextitnet.py
    """
    def __init__(self,
                item_embed_size,
                item_size=None,
                for_critic=False,
                num_blocks=1,
                dilations=[1,2,4],
                kernel_size=3,
                # dropout_rate=0.25,
        ):
        super(NextItEncoder, self).__init__()
        # common parameters
        self.item_embed_size = item_embed_size
        # self.embedding_dropout = nn.Dropout(dropout_rate)

        self.residual_channels = item_embed_size
        self.block_num = num_blocks
        self.dilations = dilations * self.block_num
        self.kernel_size = kernel_size
        
        # residual blocks    dilations in blocks:[1,2,4,8,1,2,4,8,...]
        rb = [
            ResidualBlock_b(
                self.residual_channels,
                self.residual_channels,
                kernel_size=self.kernel_size,
                dilation=dilation,
            )
            for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)
        # fully-connected layer
        self.final_layer = nn.Linear(self.residual_channels, self.item_embed_size)
        # self.final_layer.weight.data.normal_(0.0, 0.01)  # initializer
        # self.final_layer.bias.data.fill_(0.1)
        
        if not for_critic:
            self.item_size = item_size # 0 for padding
            self.item_embedding = nn.Embedding(self.item_size, self.item_embed_size, padding_idx=0)
            self.init_item_embedding()

    def init_item_embedding(self):
        std = 1. / math.sqrt(self.item_embed_size)
        self.item_embedding.weight.data.uniform_(-std, std)
    
    def modify_padding(self, seq_items, seq_length):
        new_seq_items = torch.zeros_like(seq_items)
        for i in range(seq_items.shape[0]):
            l = seq_length[i]
            new_seq_items[i, -l: ] = seq_items[i, :l]
        return new_seq_items

    def get_sess_embed(self, seq_items, seq_feedbacks, seq_length):
        seq_items = self.modify_padding(seq_items, seq_length)
        return self.item_embedding(seq_items)

    def forward_with_embed(self, sess_embed, seq_length=None):
        # Residual locks
        dilate_outputs = self.residual_blocks(sess_embed)
        hidden = dilate_outputs[:, -1, :].view(
            -1, self.residual_channels
        )  # [batch_size, embed_size]
        sess_dense = self.final_layer(hidden)  # [batch_size, embedding_size]
        return hidden

    def forward(self, seq_items, seq_feedbacks, seq_length, return_embed=False):
        sess_embed = self.get_sess_embed(seq_items, seq_feedbacks, seq_length)
        # seq_feedback_embedding = self.feedback_embedding(seq_feedbacks)
        # seq_input_embedding = torch.cat((seq_item_embedding, seq_feedback_embedding), 2)
        # [batch_size, seq_len, embed_size]
        
        sess_dense = self.forward_with_embed(sess_embed)
        if return_embed:
            return sess_embed, sess_dense
        return sess_dense
