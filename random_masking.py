# @title random_masking
import torch

def random_masking(length, mask_ratio, b=64):
    noise = torch.rand(b, length)
    len_mask = int(length * mask_ratio)
    _, msk_ind = torch.topk(noise, k=len_mask, dim=-1, sorted=False) # val, ind -> [b,len_mask]
    _, keep_ind = torch.topk(noise, k=length-len_mask, largest=False, dim=-1, sorted=False) # val, ind -> [b,len_keep]
    return msk_ind, keep_ind

# msk_ind, keep_ind = random_masking(10, .3, b=2)

# x_ = torch.rand(4, 3, 2)
# print(x_)
# # ids = torch.tensor([0, 2, 1])[None,:,None]
# # ids = torch.tensor([0, 2, 1])[None,:,None].repeat(4,1,2)
# ids = torch.tensor([1, 2, 0])[None,:,None].repeat(4,1,2)
# # o = torch.gather(x_, dim=1, index=ids)
# o = torch.zeros_like(x_).scatter_(dim=1, index=ids, src=x_)
# print(o)
