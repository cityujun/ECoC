from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from .ecoc import C3ImitTrainer


class C3ImitGraphTrainer(C3ImitTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def train_epoch(self, debug=False, eval_dataloader=None):
        self.policy.train()
        training_losses = []
        training_steps = len(self.dataloader.dataset) // self.dataloader.batch_size

        for ii, (cur_tuple, next_tuple, target_tuple) in enumerate(self.dataloader):
            alias_inputs, A, items, mask = cur_tuple
            A, items, mask = A.to(self.device), items.to(self.device), mask.to(self.device)
            encoder_inputs = (alias_inputs, A, items, mask)

            next_alias_inputs, next_A, next_items, next_mask = next_tuple
            next_A, next_items, next_mask = next_A.to(self.device), next_items.to(self.device), next_mask.to(self.device)
            next_gnn_tuple = (next_alias_inputs, next_A, next_items, next_mask)

            target_item, target_fb, terminals = target_tuple
            target_item, target_fb = target_item.to(self.device), target_fb.to(self.device)
            
            # Critic training
            cur_q1, cur_q2, next_q, next_logprob, neg_q1, neg_q2 = self.policy.critic_forward( \
                                                        encoder_inputs, next_gnn_tuple, target_item)
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
            
            # actor training, with supervised regularization
            logits, sampled_q_probs, sampled_idx, action_q = self.policy.actor_forward(encoder_inputs)
            ranking_loss = self.ce_loss(logits, target_item)            
            policy_loss = self.ce_loss(logits.gather(1, sampled_idx), sampled_q_probs)

            self.actor_optim.zero_grad()
            (ranking_loss + self.beta * policy_loss + self.alpha * action_q).backward()
            self.actor_optim.step()
            
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
    
    def eval_epoch(self):
        self.policy.eval()
        eval_losses = []
        eval_steps = len(self.valid_dataloader.dataset) // self.valid_dataloader.batch_size

        with torch.no_grad():
            for ii, (gnn_tuple, _, target_tuple) in enumerate(self.valid_dataloader):
                alias_inputs, A, items, mask = gnn_tuple
                encoder_inputs = (alias_inputs, A.to(self.device), items.to(self.device), mask.to(self.device))
                target_item, target_fb, _ = target_tuple

                # logits = self.policy.act(encoder_inputs)
                logits, sampled_q_probs, sampled_idx, action_q = self.policy.actor_forward(encoder_inputs)
                ranking_loss = self.ce_loss(logits, target_item.to(self.device)).item()
                policy_loss = self.ce_loss(logits.gather(1, sampled_idx), sampled_q_probs).item()

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

        for ii, (gnn_tuple, _, target_tuple) in enumerate(dataloader):
            alias_inputs, A, items, mask = gnn_tuple
            encoder_inputs = (alias_inputs, A.to(self.device), items.to(self.device), mask.to(self.device))
            
            target_item, target_fb, _ = target_tuple
            target_item, target_fb = target_item.to(self.device), target_fb.to(self.device)

            if debug:
                # actions_tuple, logits_tuple = self.policy.act(encoder_inputs, debug=True, return_dense=True)
                deb_logits, logits = self.policy.act(encoder_inputs, debug=True, return_dense=False)
                batch_recalls_deb, batch_mrrs_deb, batch_ndcg_deb, _ = self.compute_eval_metrics( \
                                                                            deb_logits, target_item, topk)
                total_recalls_deb += batch_recalls_deb
                total_mrrs_deb += batch_mrrs_deb
                total_ndcg_deb += batch_ndcg_deb
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
