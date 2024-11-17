import torch
from torch import nn


class BasePolicy(nn.Module):
    def __init__(self, state_encoder):
        super(BasePolicy, self).__init__()
        self.actor = state_encoder
        self.action_size = state_encoder.item_embed_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.critic = None

    def act(self, encoder_inputs):
        ## default, for BC policy
        with torch.no_grad():
            return self.forward(encoder_inputs)

    def forward(self, encoder_inputs):
        ## default, for BC policy
        return self.actor(*encoder_inputs)
    
    '''The following functions for RL policies
    '''
    def compute_logits(self, dense_outputs):
        item_weights = self.actor.item_embedding.weight
        return torch.matmul(dense_outputs, item_weights.transpose(1, 0))
    
    def get_next_tuple(self, seq_items, seq_feedbacks, seq_length, next_item, next_fb):
        batch_size = seq_items.shape[0]
        padding_tensor = torch.zeros((batch_size, 1)).long().to(self.device)
        seq_items = torch.cat([seq_items, padding_tensor], 1)
        seq_feedbacks = torch.cat([seq_feedbacks, padding_tensor], 1)
        seq_items[torch.arange(batch_size), seq_length] = next_item.view(-1)
        seq_feedbacks[torch.arange(batch_size), seq_length] = next_fb.view(-1)
        return (seq_items, seq_feedbacks, seq_length + 1)
    
    def soft_update(self, tau):
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def frozen_target_grads(self):
        for p in self.target_critic.parameters():
            p.requires_grad = False
        
        for p in self.target_actor.parameters():
            p.requires_grad = False
