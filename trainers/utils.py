import numpy as np
# import torch


def get_params_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('*' * 50)
    print(f'Total: {total_num}, Trainable: {trainable_num}'.center(50))
    print('*' * 50)


# def get_next_tuple(seq_items, seq_feedbacks, seq_length, next_item, next_fb):
#         batch_size = seq_items.shape[0]
#         padding_tensor = torch.zeros((batch_size, 1)).long().to(seq_items.device)
#         seq_items = torch.cat([seq_items, padding_tensor], 1)
#         seq_feedbacks = torch.cat([seq_feedbacks, padding_tensor], 1)
#         seq_items[torch.arange(batch_size), seq_length] = next_item.view(-1)
#         seq_feedbacks[torch.arange(batch_size), seq_length] = next_fb.view(-1)
#         return (seq_items, seq_feedbacks, seq_length + 1)


def convert_session_graph(inputs):
    '''
    return:
        alias_inputs: list of list, tensor shape: (bs,  max_batch_seq_length)
        A: list of matrix, tensor shape: (bs, max_unique_session_item, max_unique_session_item * 2)
        items: padding unique nodes, tesor shape: (bs, max_unique_session_item)
    '''
    items, n_node, A, alias_inputs = [], [], [], []
    for u_input in inputs:
        n_node.append(len(np.unique(u_input)))
    max_n_node = np.max(n_node)
    for u_input in inputs:
        node = np.unique(u_input)
        items.append(node.tolist() + (max_n_node - len(node)) * [0])
        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            if u_input[i + 1] == 0:
                break
            u = np.where(node == u_input[i])[0][0] # index in node
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1
        # normalize
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        A.append(u_A)
        alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
    # return alias_inputs, A, items
    return np.array(alias_inputs), np.array(A), np.array(items)
