import numpy as np
import torch


def get_recall(indices, targets): #recall --> wether next item in session is within top K=20 recommended items or not
    """
    Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """
    assert indices.size(0) == targets.size(0)
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero(as_tuple=False)
    if len(hits) == 0:
        return 0
    n_hits = hits.size(0)
    # recall = float(n_hits) / targets.size(0)
    return n_hits


def get_mrr(indices, targets): #Mean Receiprocal Rank --> Average of rank of next item in the session.
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        mrr (float): the mrr score
    """
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero(as_tuple=False) # (targets.size(0), 2)
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).item()
    return mrr


def get_ndcg(indices, targets):
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero(as_tuple=False) # (targets.size(0), 2)
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    weights = torch.log2(ranks + 1.)
    ndcg = torch.sum(1 / weights).item() # idcg=1
    return ndcg
