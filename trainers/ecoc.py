from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from .base_trainer import BaseRLTrainer


class C3ImitTrainer(BaseRLTrainer):
    def __init__(self,
                 alpha,
                 beta,
                 eta,
                 lr=1e-3,
                 l2=1e-5,
                 gamma=0.99,
                 tau=0.005,
                 *args,
                 **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.actor_optim = torch.optim.Adam(self.policy.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.policy.critic.parameters(), lr=lr, weight_decay=l2)
        self.n_epoch = 0
        
    def train_epoch(self, debug=False, eval_dataloader=None):
        self.policy.train()
        training_losses = []
        training_steps = len(self.dataloader.dataset) // self.dataloader.batch_size

        for ii, (encoder_inputs, target_tuple, terminals) in enumerate(self.dataloader):#, total=training_steps):
            seq_items, seq_actions, seq_length = encoder_inputs
            target_item, target_fb = target_tuple

            seq_items, seq_actions = seq_items.to(self.device), seq_actions.to(self.device)
            target_item, target_fb = target_item.to(self.device), target_fb.to(self.device)
            encoder_inputs = (seq_items, seq_actions, seq_length)
            
            ## Critic training
            cur_q1, cur_q2, next_q, next_logprob, neg_q1, neg_q2 = self.policy.critic_forward( \
                                                        seq_items, seq_actions, seq_length, target_item, target_fb)
            rewards = (target_fb >= 1).float().view(-1, 1) * 0.5 - 1.
            # target_probs = self.env.get_rewards(seq_items, seq_actions, seq_length, actions, target_item)
            # rewards = target_probs.float().view(-1, 1).to(self.device) # better than 0/1 rewards
            assert rewards.shape == next_q.shape
            target_q = rewards + self.gamma * next_q * terminals.view(-1, 1).to(self.device)
            td_loss = self.mse_loss(cur_q1, target_q) + self.mse_loss(cur_q2, target_q)

            assert cur_q1.shape[1] == 1
            ## bpr loss
            reg_loss_1 = - F.logsigmoid(cur_q1.expand_as(neg_q1) - neg_q1).mean() - \
                        F.logsigmoid(cur_q2.expand_as(neg_q2) - neg_q2).mean()
            ## additional for vanilla implementation
            neg_q1 = torch.softmax(neg_q1, dim=-1) * neg_q1
            neg_q2 = torch.softmax(neg_q2, dim=-1) * neg_q2
            reg_loss_2 = (neg_q1.sum(dim=-1) - cur_q1.view(-1)).mean() + (neg_q2.sum(dim=-1) - cur_q2.view(-1)).mean()

            self.critic_optim.zero_grad()
            (td_loss + self.eta * reg_loss_2).backward()
            self.critic_optim.step()
            
            ## actor training, with supervised regularization
            logits, sampled_q_probs, sampled_idx, action_q = self.policy.actor_forward(encoder_inputs)
            ranking_loss = self.ce_loss(logits, target_item)
            policy_loss = self.ce_loss(logits.gather(1, sampled_idx), sampled_q_probs)

            self.actor_optim.zero_grad()
            # (ranking_loss + self.beta[self.n_epoch] * policy_loss + self.alpha[self.n_epoch] * action_q).backward()
            (ranking_loss + self.beta * policy_loss + self.alpha * action_q).backward()
            self.actor_optim.step()
            
            # losses.append((ranking_loss.item(), actor_loss.item(), critic_loss.item(), cur_q1.mean().item(), target_q.mean().item()))
            training_losses.append([ranking_loss.item(), policy_loss.item(), action_q.item()] + \
                                    [cur_q1.mean().item(), target_q.mean().item(), next_logprob.mean().item()] + \
                                    [td_loss.item(), reg_loss_1.item(), reg_loss_2.item()])

            self.policy.soft_update(self.tau)

        mean_training_losses = np.around(np.mean(training_losses, axis=0), 4)
        eval_steps, mean_eval_loss = self.eval_epoch()
        print("************ Staring print *********************")
        print(f'Current alpha: {self.alpha}, beta: {self.beta}, eta: {self.eta}')
        self.total_train_steps += training_steps
        self.n_epoch += 1
        return training_steps, eval_steps, mean_training_losses, mean_eval_loss
    
    def print_training_info(self, epoch_id, interval, n_steps, losses):
        # print(f'Epoch {epoch_id}, time: {interval}, total steps: {n_steps}, ranking loss: {losses}')
        print(f'Epoch {epoch_id}, time: {interval}, total steps: {n_steps}, ' + \
                f'ranking loss: {losses[0]}, policy loss: {losses[1]},  q loss: {losses[2]}, ' + \
                f'cur_q1: {losses[3]}, tar_q: {losses[4]}, log prob: {losses[5]}, ' + \
                f'td loss: {losses[6]}, bpr reg loss: {losses[7]}, vnl reg loss: {losses[8]}')
    
    def eval_epoch(self):
        self.policy.eval()
        eval_losses = []
        eval_steps = len(self.valid_dataloader.dataset) // self.valid_dataloader.batch_size

        with torch.no_grad():
            for ii, (encoder_inputs, target_tuple, terminals) in enumerate(self.valid_dataloader):
                seq_items, seq_actions, seq_length = encoder_inputs
                target_item, target_fb = target_tuple

                encoder_inputs = (seq_items.to(self.device), seq_actions.to(self.device), seq_length)
                # logits = self.policy.act(encoder_inputs)
                logits, sampled_q_probs, sampled_idx, action_q = self.policy.actor_forward(encoder_inputs)
                ranking_loss = self.ce_loss(logits, target_item.to(self.device)).item()
                policy_loss = self.ce_loss(logits.gather(1, sampled_idx), sampled_q_probs).item()
                # policy_loss = self.ce_loss(logits[:, 1:], sampled_q_probs).item()

                # eval_losses.append(ranking_loss.item())
                total_loss = ranking_loss + self.beta * policy_loss + self.alpha * action_q.item()
                eval_losses.append([ranking_loss, policy_loss, action_q.item(), total_loss])

        mean_eval_loss = np.around(np.mean(eval_losses, axis=0), 4)
        return eval_steps, mean_eval_loss

    def test(self, dataloader, topk, debug=False):
        self.policy.eval()
        total_recalls, total_mrrs, total_ndcg, total_nums = 0, 0, 0, 0
        total_recalls_deb, total_mrrs_deb, total_ndcg_deb = 0, 0, 0
        total_reward, total_reward_deb = 0, 0
        eval_steps = len(dataloader.dataset) // dataloader.batch_size

        for ii, (encoder_inputs, target_tuple, _) in enumerate(dataloader): #, total=eval_steps):
            seq_items, seq_actions, seq_length = encoder_inputs
            target_item, target_fb = target_tuple

            target_item, target_fb = target_item.to(self.device), target_fb.to(self.device)
            encoder_inputs = (seq_items.to(self.device), seq_actions.to(self.device), seq_length)

            if debug:
                pass
            else:
                logits = self.policy.act(encoder_inputs, debug=False)

            batch_recalls, batch_mrrs, batch_ndcg, batch_nums = self.compute_eval_metrics( \
                                                                        logits, target_item, topk)
            total_recalls += batch_recalls
            total_mrrs += batch_mrrs
            total_ndcg += batch_ndcg
            total_nums += batch_nums

        self.total_eval_steps += eval_steps

        rec, mrr, ndcg = round(total_recalls / total_nums, 4), round(total_mrrs / total_nums, 4), round(total_ndcg / total_nums, 4)
        
        if debug:
            rec_deb, mrr_deb, ndcg_deb = round(total_recalls_deb/total_nums, 4), round(total_mrrs_deb/total_nums, 4), round(total_ndcg_deb/total_nums, 4)
            return (rec_deb, rec), (mrr_deb, mrr), (ndcg_deb, ndcg)
        return rec, mrr, ndcg
