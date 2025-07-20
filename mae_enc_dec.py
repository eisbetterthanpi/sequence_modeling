# @title mae enc dec
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x): # [b,d,t]
        b,d,t = x.shape
        pad = (self.patch_size - t % self.patch_size) % self.patch_size
        x = F.pad(x, (0,pad))
        return self.proj(x)


class Encoder(nn.Module):
    # def __init__(self, in_dim, d_model, out_dim=None, nhead=8, d_hid=None, nlayers=1, drop=0):
    def __init__(self, patch_size, in_dim, d_model, out_dim=None, n_heads=4, nlayers=1, drop=0):
        super().__init__()
        # act = nn.ReLU() # ReLU SiLU GELU
        self.embed = PatchEmbed(patch_size, in_dim, d_model)
        # self.pos_emb = nn.Parameter(torch.randn(1, 200, d_model)*.02)
        self.pos_emb = RoPE(d_model, seq_len=500, base=10000)
        self.transformer = nn.Sequential(*[AttentionBlock(d_model, n_heads=n_heads) for _ in range(nlayers)])
        self.norm = nn.LayerNorm(d_model) # LayerNorm RMSNorm
        # self.lin = nn.Linear(d_model, out_dim) if out_dim and out_dim != d_model else None

    def forward(self, x, mask_indices=None): # [b,t,in], [batch, num_context_toks] # True will be ignored by the attention # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        # print("mae enc fwd",x.shape)
        x = self.embed(x.transpose(-2,-1)).transpose(-2,-1) # [b,t,d]
        b,t = x.shape[:2]
        x = x + self.pos_emb[:,:t]
        if mask_indices != None: x = x[torch.arange(b).unsqueeze(-1), mask_indices] # [batch, num_context_toks, d_model]
        x = self.transformer(x)
        out = self.norm(x)
        # if self.lin: out = self.lin(out)
        return out

class Decoder(nn.Module):
    def __init__(self, in_dim, d_model, out_dim=None, n_heads=4, nlayers=1, drop=0):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)# if in_dim != d_model else None
        # self.pos_emb = nn.Parameter(torch.randn(1, 200, d_model)*.02)
        self.pos_emb = RoPE(d_model, seq_len=200, base=10000)
        self.transformer = nn.Sequential(*[AttentionBlock(d_model, n_heads=n_heads) for _ in range(nlayers)])
        self.cls = nn.Parameter(torch.randn(1,1,d_model)*0.02) # randn zeros
        self.norm = nn.LayerNorm(d_model) # LayerNorm RMSNorm
        self.lin = nn.Linear(d_model, out_dim or d_model)

    def forward(self, x, mask_indices, trg_indices): # [b,m,d], [b,m], [b,t-m]
        x = self.embed(x) # [b,m,d]
        ids_restore = torch.cat([mask_indices, trg_indices], dim=-1).unsqueeze(-1).repeat(1,1,x.shape[-1]) # [b,t]

        x = torch.cat([x, self.cls.repeat(x.shape[0],trg_indices.shape[1],1)], dim=1) # [b,m+(t-m),d]
        # print("Trans pred",x.shape, ids_restore.shape)
        x = torch.zeros_like(x, device=device).scatter_(dim=1, index=ids_restore.to(device), src=x) # unshuffle # The backward pass is implemented only for src.shape == index.shape
        # x = torch.scatter(torch.zeros_like(x), dim=1, index=ids, src=x) # unshuffle
        x = x + self.pos_emb[0,:x.shape[1]]

        out = self.transformer(x)
        out = self.norm(out)
        out = self.lin(out)
        return out # [b,t,d]


# batch, seq_len, d_model = 4,3500,16
# in_dim = 3
# patch_size=32
# model = Encoder(patch_size, in_dim, d_model, n_heads=4, nlayers=1, drop=0.).to(device)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad)) # 27584
# x =  torch.rand((batch, seq_len, in_dim), device=device)
# out = model(x)
# print(out.shape)
# # # print(out)
# model = Decoder(in_dim, d_model, out_dim=None, d_head=4, nlayers=1).to(device)
# out = model(out)
# print(out.shape)

