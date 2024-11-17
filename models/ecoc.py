import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from .base_policy import BasePolicy
from .critic import GenericCritic


class C3ImitDuplicatedPolicy(BasePolicy):
    ''' v5, final version of c3 imitator,
    Euclidean version, supervised regularization, plus q value gradient ascent
    Utlized two same state encoder as a critic, Q value is (critic_encoder_output * MLP(action)).sum()
    for non-GNN backbone
    '''
    def __init__(self,
                 critic_encoder_1,
                 critic_encoder_2,
                 lmbda,
                 dist_sample_k,
                 *args,
                 **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.critic = GenericCritic(critic_encoder_1, critic_encoder_2, self.action_size)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_actor = copy.deepcopy(self.actor)
        self.frozen_target_grads()
        
        self.lmbda = lmbda
        self.dist_sample_k = dist_sample_k
        self = self.to(self.device)

    def act(self, encoder_inputs, debug=False, return_dense=False):
        with torch.no_grad():
            target_actions = self.target_actor(*encoder_inputs)
            target_item_weights = self.target_actor.item_embedding.weight.data
            target_logits = torch.matmul(target_actions, target_item_weights.transpose(1, 0))
        if debug:
            with torch.no_grad():
                actions = self.actor(*encoder_inputs)
                logits = self.compute_logits(actions)
            if return_dense:
                return (actions, target_actions), (logits, target_logits)
            return logits, target_logits
        return target_logits

    def critic_forward(self, seq_items, seq_feedbacks, seq_length, next_item, next_fb):
        with torch.no_grad():
            sess_embed, actor_actions = self.actor(seq_items, seq_feedbacks, seq_length, return_embed=True)
        item_weights = self.actor.item_embedding.weight.data.clone().detach()
        actions = item_weights[ next_item.view(-1) ]
        cur_q1, cur_q2, q1_vec, q2_vec = self.critic(sess_embed, seq_length, actions, return_q_vec=True)

        ## add regularizer
        cur_logits = torch.matmul(actor_actions, item_weights.transpose(1, 0)).detach()
        
        ## v1, better, align as below
        cur_probs = torch.softmax(cur_logits, dim=1)
        cur_sampled_indices = torch.multinomial(cur_probs, self.dist_sample_k, replacement=False)
        cur_neg_actions = item_weights[ cur_sampled_indices.reshape(-1) ] # (bs, 500-1, embed_size)
        ## v2
        # _, cur_top_indices = torch.topk(cur_logits, dim=-1, k=self.dist_sample_k)
        # cur_neg_actions = item_weights[ cur_top_indices.reshape(-1) ]
        
        repeated_q1_vec = torch.repeat_interleave(q1_vec, self.dist_sample_k, dim=0)
        repeated_q2_vec = torch.repeat_interleave(q2_vec, self.dist_sample_k, dim=0)
        cur_neg_q1 = self.critic.min_q_with_q_vec(repeated_q1_vec, cur_neg_actions).view(-1, self.dist_sample_k)
        cur_neg_q2 = self.critic.min_q_with_q_vec(repeated_q2_vec, cur_neg_actions).view(-1, self.dist_sample_k)
        
        ## next_obs
        next_tuple = self.get_next_tuple(seq_items, seq_feedbacks, seq_length, next_item, next_fb)

        ## regularization for current q values, not used
        # with torch.no_grad():
        #     next_sess_embed, next_dense_actions = self.actor(*next_tuple, return_embed=True)
        #     next_logits = torch.matmul(next_dense_actions, item_weights.transpose(1, 0))
        #     next_probs = torch.softmax(next_logits, dim=1)
        #     next_sampled_indices = torch.multinomial(next_probs, self.dist_sample_k, replacement=False)
        #     next_neg_actions = item_weights[ next_sampled_indices.reshape(-1) ]

        # next_q1_vec, next_q2_vec = self.critic.get_q_tuple(next_sess_embed, seq_length+1)
        # repeated_next_q1_vec = torch.repeat_interleave(next_q1_vec, self.dist_sample_k, dim=0)
        # repeated_next_q2_vec = torch.repeat_interleave(next_q2_vec, self.dist_sample_k, dim=0)
        # next_neg_q1 = self.critic.min_q_with_q_vec(repeated_next_q1_vec, next_neg_actions).view(-1, self.dist_sample_k)
        # next_neg_q2 = self.critic.min_q_with_q_vec(repeated_next_q2_vec, next_neg_actions).view(-1, self.dist_sample_k)
        
        with torch.no_grad():
            next_sess_embed, next_dense_actions = self.target_actor(*next_tuple, return_embed=True)
            target_item_weights = self.target_actor.item_embedding.weight.data
            next_logits = torch.matmul(next_dense_actions, target_item_weights.transpose(1, 0))
            
            ## v1
            next_dist = Categorical(logits=next_logits)
            next_action_idx = next_dist.sample() # (-1, )
            next_logprob = next_dist.log_prob(next_action_idx)
            # [TODO], better without target
            next_actions = target_item_weights[ next_action_idx.long() ]
            ## v2
            # next_top_logits, next_top_indices = torch.topk(next_logits, dim=-1, k=self.dist_sample_k)
            # next_top_dist = Categorical(logits=next_top_logits)
            # next_action_idx = next_top_dist.sample() # (-1, )
            # next_logprob = next_top_dist.log_prob(next_action_idx)
            # next_actions = target_item_weights[ next_top_indices.gather(1, next_action_idx.long().view(-1, 1)).view(-1) ]

            next_q = self.target_critic.min_q(next_sess_embed, seq_length+1, next_actions)
            next_q = next_q - self.lmbda * next_logprob.view(-1, 1)

        return cur_q1, cur_q2, next_q, next_logprob, cur_neg_q1, cur_neg_q2

    def actor_forward(self, encoder_inputs):
        seq_length = encoder_inputs[-1]
        sess_embed, actions = self.actor(*encoder_inputs, return_embed=True)
        critic_q_vec = self.critic.get_q_vec(sess_embed, seq_length).detach()
        logits = self.compute_logits(actions)
        item_weights = self.actor.item_embedding.weight.data
        
        with torch.no_grad():
            probs = torch.softmax(logits.detach(), dim=1)
            sampled_action_idx = torch.multinomial(probs, self.dist_sample_k, replacement=False)
            repeated_q_vec = torch.repeat_interleave(critic_q_vec, self.dist_sample_k, dim=0)
            repeated_actions = item_weights[ sampled_action_idx.view(-1) ]
            part_q_vals = self.critic.min_q_with_q_vec(repeated_q_vec, repeated_actions).view(-1, self.dist_sample_k)
            part_q_probs = torch.softmax(part_q_vals, dim=-1)
        
        return logits, part_q_probs, sampled_action_idx, \
                        - self.critic.min_q_with_q_vec(critic_q_vec, actions).mean()


LONG = torch.LongTensor
FLOAT = torch.FloatTensor

class C3ImitGraphPolicy(C3ImitDuplicatedPolicy):
    """ v5, final version of c3 imitator,
    Euclidean version, supervised regularization, plus q value gradient ascent
    for GNN-based backbone only
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_sess_tuple(gnn_tuple):
        alias_inputs, A, _, mask = gnn_tuple
        return (alias_inputs, A, mask)

    def critic_forward(self, cur_gnn_tuple, next_gnn_tuple, next_item):
        sess_tuple = self.get_sess_tuple(cur_gnn_tuple)
        with torch.no_grad():
            sess_embed, actor_actions = self.actor(*cur_gnn_tuple, return_embed=True)
        item_weights = self.actor.item_embedding.weight.data.clone().detach()
        actions = item_weights[ next_item.view(-1) ]
        cur_q1, cur_q2, q1_vec, q2_vec = self.critic(sess_embed, sess_tuple, actions, return_q_vec=True)
        
        ## add regularizer
        cur_logits = torch.matmul(actor_actions, item_weights.transpose(1, 0)).detach()
        ## v1, better, align as below
        cur_probs = torch.softmax(cur_logits, dim=1)
        cur_sampled_indices = torch.multinomial(cur_probs, self.dist_sample_k, replacement=False)
        cur_neg_actions = item_weights[ cur_sampled_indices.reshape(-1) ] # (bs, 500-1, embed_size)
        ## v2
        # _, cur_top_indices = torch.topk(cur_logits, dim=-1, k=self.dist_sample_k)
        # cur_neg_actions = item_weights[ cur_top_indices.reshape(-1) ]
        repeated_q1_vec = torch.repeat_interleave(q1_vec, self.dist_sample_k, dim=0)
        repeated_q2_vec = torch.repeat_interleave(q2_vec, self.dist_sample_k, dim=0)
        cur_neg_q1 = self.critic.min_q_with_q_vec(repeated_q1_vec, cur_neg_actions).view(-1, self.dist_sample_k)
        cur_neg_q2 = self.critic.min_q_with_q_vec(repeated_q2_vec, cur_neg_actions).view(-1, self.dist_sample_k)

        # next obs, need extra conversion
        # next_seq_items, next_seq_mask, _ = self.get_next_tuple(seq_items, seq_mask, seq_length, next_item, torch.ones_like(next_item))
        # alias_inputs, A, items = convert_session_graph(next_seq_items.cpu().numpy())
        # next_gnn_tuple = (LONG(alias_inputs), FLOAT(A).to(self.device), LONG(items).to(self.device), next_seq_mask)
        # next_sess_tuple = (LONG(alias_inputs), FLOAT(A).to(self.device), next_seq_mask)
        next_sess_tuple = self.get_sess_tuple(next_gnn_tuple)
        with torch.no_grad():
            next_sess_embed, next_actor_actions = self.target_actor(*next_gnn_tuple, return_embed=True)

            target_item_weights = self.target_actor.item_embedding.weight.data
            next_logits = torch.matmul(next_actor_actions, target_item_weights.transpose(1, 0))
            
            ## v1
            next_dist = Categorical(logits=next_logits)
            next_action_idx = next_dist.sample() # (-1, )
            next_logprob = next_dist.log_prob(next_action_idx)
            next_actions = target_item_weights[ next_action_idx.long() ]
            ## v2
            # next_top_logits, next_top_indices = torch.topk(next_logits, dim=-1, k=self.dist_sample_k)
            # next_top_dist = Categorical(logits=next_top_logits)
            # next_action_idx = next_top_dist.sample() # (-1, )
            # next_logprob = next_top_dist.log_prob(next_action_idx)
            # next_actions = target_item_weights[ next_top_indices.gather(1, next_action_idx.long().view(-1, 1)).view(-1) ]

            next_q = self.target_critic.min_q(next_sess_embed, next_sess_tuple, next_actions)
            next_q = next_q - self.lmbda * next_logprob.view(-1, 1)

        return cur_q1, cur_q2, next_q, next_logprob, cur_neg_q1, cur_neg_q2

    def actor_forward(self, encoder_inputs):
        sess_tuple = self.get_sess_tuple(encoder_inputs)
        sess_embed, actions = self.actor(*encoder_inputs, return_embed=True)
        critic_q_vec = self.critic.get_q_vec(sess_embed, sess_tuple).detach()
        logits = self.compute_logits(actions)
        item_weights = self.actor.item_embedding.weight.data

        with torch.no_grad():
            probs = torch.softmax(logits.detach(), dim=1)
            sampled_action_idx = torch.multinomial(probs, self.dist_sample_k, replacement=False)
            repeated_q_vec = torch.repeat_interleave(critic_q_vec, self.dist_sample_k, dim=0)
            repeated_actions = item_weights[ sampled_action_idx.view(-1) ]
            part_q_vals = self.critic.min_q_with_q_vec(repeated_q_vec, repeated_actions).view(-1, self.dist_sample_k)
            part_q_probs = torch.softmax(part_q_vals, dim=-1)
        
        return logits, part_q_probs, sampled_action_idx, \
                        - self.critic.min_q_with_q_vec(critic_q_vec, actions).mean()
