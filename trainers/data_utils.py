import torch
from .utils import convert_session_graph

LONG = torch.LongTensor
FLOAT = torch.FloatTensor

def rnn_collate_wrapper(batch, positive_threshold=1):
    batch_seq_items, batch_seq_feedbacks = [], []
    batch_targets, batch_labels = [], []
    batch_terminals = []
    for seq_items, seq_feedbacks, target, label, terminal in batch:
        batch_seq_items.append(seq_items)
        batch_seq_feedbacks.append(seq_feedbacks)
        batch_targets.append(target)
        batch_labels.append(label)
        batch_terminals.append(terminal)
    
    # padding
    seq_length = [len(elem) for elem in batch_seq_items]
    max_batch_length = max(seq_length)
    batch_seq_items = [elem + [0] * (max_batch_length-len(elem)) for elem in batch_seq_items]
    batch_seq_feedbacks = [elem + [0] * (max_batch_length-len(elem)) for elem in batch_seq_feedbacks]
    # batch_labels = [feedback >= positive_threshold for feedback in batch_labels]
    
    return (LONG(batch_seq_items), LONG(batch_seq_feedbacks), LONG(seq_length)), \
            (LONG(batch_targets), LONG(batch_labels)), \
            FLOAT(batch_terminals) #FLOAT(batch_labels)


def gnn_collate_wrapper(batch, positive_threshold=1):
    batch_seq_items, batch_seq_feedbacks = [], []
    batch_targets, batch_labels = [], []
    for seq_items, seq_feedbacks, target, label, _ in batch:
        batch_seq_items.append(seq_items)
        batch_seq_feedbacks.append(seq_feedbacks)
        batch_targets.append(target)
        batch_labels.append(label)
    
    # padding
    seq_length = [len(elem) for elem in batch_seq_items]
    max_batch_length = max(seq_length)
    masks = [[1] * elem + [0] * (max_batch_length - elem) for elem in seq_length]
    batch_seq_items = [elem + [0] * (max_batch_length-len(elem)) for elem in batch_seq_items]
    batch_seq_feedbacks = [elem + [0] * (max_batch_length-len(elem)) for elem in batch_seq_feedbacks]
    # batch_labels = [feedback >= positive_threshold for feedback in batch_labels]
    
    alias_inputs, A, items = convert_session_graph(batch_seq_items)
    
    return (LONG(alias_inputs), FLOAT(A), LONG(items), LONG(masks)), \
            (LONG(batch_targets), LONG(batch_labels)), \
            FLOAT(batch_labels)


def gnn_with_next_collate_wrapper(batch, positive_threshold=1):
    batch_seq_items, batch_seq_feedbacks = [], []
    batch_next_seq_items = []
    batch_targets, batch_labels = [], []
    batch_terminals = []
    
    for seq_items, seq_feedbacks, target, label, terminal in batch:
        batch_seq_items.append(seq_items)
        batch_next_seq_items.append(seq_items + [target])
        batch_seq_feedbacks.append(seq_feedbacks)
        batch_targets.append(target)
        batch_labels.append(label)
        batch_terminals.append(terminal)
    
    # padding
    seq_length = [len(elem) for elem in batch_seq_items]
    max_batch_length = max(seq_length)
    batch_seq_items = [elem + [0] * (max_batch_length-len(elem)) for elem in batch_seq_items]
    batch_seq_feedbacks = [elem + [0] * (max_batch_length-len(elem)) for elem in batch_seq_feedbacks]
    batch_next_seq_items = [elem + [0] * (max_batch_length+1-len(elem)) for elem in batch_next_seq_items]
    # batch_labels = [feedback >= positive_threshold for feedback in batch_labels]
    
    # additional operations for gnn
    masks = [[1] * elem + [0] * (max_batch_length - elem) for elem in seq_length]
    next_masks = [[1] * (elem+1) + [0] * (max_batch_length - elem) for elem in seq_length]
    
    alias_inputs, A, items = convert_session_graph(batch_seq_items)
    next_alias_inputs, next_A, next_items = convert_session_graph(batch_next_seq_items)
    
    # return (LONG(batch_seq_items), LONG(batch_seq_feedbacks), LONG(seq_length)), \
    return (LONG(alias_inputs), FLOAT(A), LONG(items), LONG(masks)), \
            (LONG(next_alias_inputs), FLOAT(next_A), LONG(next_items), LONG(next_masks)), \
            (LONG(batch_targets), LONG(batch_labels), FLOAT(batch_terminals))
