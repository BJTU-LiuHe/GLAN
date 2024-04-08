import torch
def binaryloss(output, target, weights=None, gamma=2):
    if weights is not None:
        assert weights.size(1) == 2

        loss = (torch.pow(1.0-output, gamma)*torch.mul(target, weights[:, 1]) * torch.log(output+1e-8)) + \
               (torch.mul((1.0 - target), weights[:, 0]) * torch.log(1.0 - output+1e-8)*torch.pow(output, gamma))
    else:
        loss = target * torch.log(output+1e-8) + (1 - target) * torch.log(1 - output+1e-8)

    return torch.neg(torch.mean(loss))

def sum_loss(output,target):

    t_row_sum=torch.sum(target,dim=0).detach()
    row_sum = torch.sum(output, dim=0)
    row_sum_diff = torch.mean(torch.pow(row_sum - t_row_sum, 2))
    t_col_sum=torch.sum(target,dim=1).detach()
    col_sum = torch.sum(output, dim=1)
    col_sum_diff = torch.mean(torch.pow(col_sum - t_col_sum, 2))

    return torch.log(1+row_sum_diff)+torch.log(1+col_sum_diff)

def sparse_loss(output,target):
    t_row_norm=torch.norm(target,dim=1).detach()
    t_col_norm=torch.norm(target,dim=0).detach()
    row_norm_diff = torch.mean(torch.pow(torch.norm(output, dim=1)-t_row_norm, 2))
    col_norm_diff = torch.mean(torch.pow(torch.norm(output, dim=0)-t_col_norm, 2))

    return row_norm_diff+col_norm_diff
