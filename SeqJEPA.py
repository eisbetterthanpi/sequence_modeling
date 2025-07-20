# @title SeqJEPA
import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

class SeqJEPA(nn.Module):
    def __init__(self, in_dim=3, d_model=32, out_dim=None, nlayers=2, n_heads=4, drop=0):
        super().__init__()
        if out_dim is None: out_dim = d_model
        self.patch_size = 8 # 8 32
        self.student = TransformerModel(self.patch_size, in_dim, d_model, out_dim=out_dim, n_heads=n_heads, nlayers=nlayers, drop=drop)
        self.predicter = TransformerPredictor(out_dim, d_model//2, out_dim, n_heads=4, nlayers=1, drop=drop)
        import copy
        self.teacher = copy.deepcopy(self.student)
        self.teacher.requires_grad_(False)
        # self.transform = RandomResizedCrop1d(3500, scale=(.8,1.))

    def loss(self, x): # [batch, T, 3]
        batch, seq, dim = x.shape
        # print(x.shape)
        # target_mask = multiblock(seq//self.patch_size, min_s=.2, max_s=.3, M=4, B=1).any(1).squeeze(1) # best.2.3M4 og.15.2M4# mask out targets to be predicted # [M, seq]
        # # target_mask = randpatch(seq//self.patch_size, mask_size=8, gamma=.9).unsqueeze(0) # 8.9 [seq]

        # print(target_mask.shape, x.shape)
        # context_mask = ~multiblock(seq//self.patch_size, min_s=.85, max_s=1, M=1, B=1).squeeze(1)|target_mask # og .85,1.M1 # [1, seq], True->Mask
        # context_mask = torch.zeros((1,seq//self.patch_size), dtype=bool)|target_mask # [1,h,w], True->Mask

        # context_indices, trg_indices = simplexmask1d(seq//self.patch_size, ctx_scale=(.85,1), trg_scale=(.7,.8), B=batch, chaos=[3,.5])
        # context_indices, trg_indices = simplexmask1d(seq//self.patch_size, ctx_scale=(.8,.9), trg_scale=(.7,.8), B=batch, chaos=[1,.5])
        # context_indices, trg_indices = simplexmask1d(seq//self.patch_size, ctx_scale=(.8,1), trg_scale=(.2,.8), B=batch, chaos=[1,.5])
        # print(context_indices.shape, trg_indices.shape)

        # context_indices = context_indices.repeat(batch,1)
        # trg_indices = trg_indices.repeat(batch,1)
        # context_mask = ~context_mask|target_mask # [1,]
        # context_indices = (~context_mask).nonzero()[:,1].unsqueeze(0).repeat(batch,1)
        # # print(trg_indices.shape, context_indices.shape)
        # # print(context_mask.shape,target_mask.shape, x.shape)
        # target_mask, context_mask = target_mask.to(device), context_mask.to(device)
        # # # target_mask, context_mask = target_mask.repeat(batch,1), context_mask.repeat(batch,1)
        # # x_ = x * F.adaptive_avg_pool1d((~context_mask).float(), x.shape[1]).unsqueeze(-1) # zero masked locations

        # context_indices = (~context_mask).nonzero()[:,1].unflatten(0, (batch,-1)) # int idx [num_context_toks] , idx of context not masked
        # trg_indices = target_mask.nonzero()[:,1].unflatten(0, (batch,-1)) # int idx [num_trg_toks] , idx of targets that are masked


        # mask_collator = MaskCollator(length=seq//self.patch_size, enc_mask_scale=(.85,1), pred_mask_scale=(.15,.2), nenc=1, npred=4, min_keep=4, allow_overlap=False)
        mask_collator = MaskCollator(length=seq//self.patch_size, enc_mask_scale=(.85,1), pred_mask_scale=(.2,.25), nenc=1, npred=4, min_keep=4, allow_overlap=False)
        collated_masks_enc, collated_masks_pred = mask_collator(batch) # idx of ctx, idx of masked trg
        # # collated_masks_enc, collated_masks_pred = mask_collator(1) # idx of ctx, idx of masked trg
        context_indices, trg_indices = torch.stack(collated_masks_enc).squeeze(0), torch.stack(collated_masks_pred).transpose(0,1).flatten(1).unique(dim=1) # [num_msk, b,num_tok]->[b,num_tok] # [64, 65], [64, 32]
        # # # zero_mask = torch.zeros(batch ,seq//self.patch_size, device=device)
        # # # zero_mask[torch.arange(batch).unsqueeze(-1), context_indices] = 1
        # # zero_mask = torch.zeros(1 ,seq//self.patch_size, device=device)
        # # zero_mask[:, context_indices] = 1
        # # x_ = x * F.adaptive_avg_pool1d(zero_mask, x.shape[1]).unsqueeze(-1) # zero masked locations
        # # context_indices, trg_indices = context_indices.repeat(batch,1), trg_indices.repeat(batch,1)


        # print('x_',x_.shape, context_indices.shape, trg_indices.shape)

        sx = self.student(x, context_indices=context_indices) # [batch, num_context_toks, out_dim]
        # print('seq_jepa loss sx',sx.shape)
        sy_ = self.predicter(sx, context_indices=context_indices, trg_indices=trg_indices) # [batch*M, num_trg_toks, out_dim]
        sy_ = F.layer_norm(sy_, (sy_.size(-1),))
        with torch.no_grad():
            sy = self.teacher(x.detach()) # [batch, num_trg_toks, out_dim]
            sy = sy[torch.arange(sy.shape[0]).unsqueeze(-1), trg_indices] # [batch, num_context_toks, d_model] # nan bec len(trg_ind)==0 # print('loss sy',torch.isnan(sy).any())
            sy = F.layer_norm(sy, (sy.size(-1),))
        loss = F.mse_loss(sy, sy_)
        return loss

    def forward(self, x): # [batch, T, 3]
        sx = self.student(x)
        out = sx.mean(dim=1)
        return out

# min_s=0.15, max_s, M
# trg.15.2M4 C.85 1

# 1e-2,1e-3 < 3e-3,1e-3
# patch16 < patch32
# NoPE good but sus

# ctx/trg sacle min/max, num blk,

in_dim = X[0].shape[-1] # 3
out_dim = train_data.vocab_size # 16
print(in_dim, out_dim)
d_model=64
# seq_jepa = SeqJEPA(in_dim=in_dim, d_model=d_model, out_dim=None, nlayers=1, n_heads=8).to(device)#.to(torch.float)
seq_jepa = SeqJEPA(in_dim=in_dim, d_model=d_model, out_dim=None, nlayers=1, n_heads=8, drop=.0).to(device)#.to(torch.float)
optim = torch.optim.AdamW(seq_jepa.parameters(), lr=1e-3) # 1e-3? default 1e-2
# optim = torch.optim.AdamW(seq_jepa.parameters(), lr=3e-4) # 1e-3? default 1e-2
# optim = StableAdamW(seq_jepa.parameters(), lr=1e-3)
# optim = Lion(seq_jepa.parameters(), lr=1e-4, weight_decay=1e-1) # lr 1e-3, wd 1e-1
# optim = torch.optim.AdamW([{'params': seq_jepa.student.parameters()},
#     {'params': seq_jepa.predicter.parameters(), 'lr': 3e-3}], lr=1e-3, weight_decay=1e-2) # default 1e-2, 5e-2
    # {'params': seq_jepa.predicter.parameters(), 'lr': 1e-2}], lr=1e-3, weight_decay=1e-2)

# !pip install -q bitsandbytes
# import bitsandbytes as bnb
# # # optim = bnb.optim.(seq_jepa.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-2)
# # optim = bnb.optim.Lion(seq_jepa.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-2) # Lion8bit
# optim = bnb.optim.Lion(seq_jepa.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-1) # Lion8bit

# https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml
# d_model 1024,384
# depth 12,6/12
# wd 5e-2 - 4e-1
# adamw 1e-4 - 1e-3 - 1e-6
# ema 0.996-1

print(sum(p.numel() for p in seq_jepa.parameters() if p.requires_grad)) # 27584
# print(sum(p.numel() for p in seq_jepa.parameters())) # 27584
# print(sum(p.numel() for p in seq_jepa.predicter.transformer_encoder.parameters() if p.requires_grad)) # 27584
# print(sum(p.numel() for p in seq_jepa.student.transformer_encoder.parameters() if p.requires_grad)) # 27584
# print(sum(p.numel() for p in seq_jepa.teacher.transformer_encoder.parameters() if p.requires_grad)) # 27584
# d_model^2 * nlayers

x = torch.rand((24, 1700, in_dim), device=device)
out = seq_jepa.loss(x)
print(out.shape)

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)
    def forward(self, x): return self.classifier(x)
classifier = Classifier(d_model, out_dim).to(device)
coptim = torch.optim.SGD(classifier.parameters(), lr=1e-3) # 1e-3
# optim = torch.optim.AdamW([{'params': seq_jepa.parameters()}, {'params': classifier.parameters(), 'lr': 1e-3}], lr=1e-3)
# coptim = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
