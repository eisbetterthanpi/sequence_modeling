# @title MAE
import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

def patchify(x, p=16): # [b,t*p,d]
    t = x.shape[1]
    pad = (p - t % p) % p
    x = F.pad(x, (0,pad,0,0))
    return x.unflatten(1, (t//p, p)).flatten(2) # [b,t,p*d]

class MAE(nn.Module):
    def __init__(self, in_dim=16, d_model=64, out_dim=None, nlayers=1, n_heads=8, drop=0):
        super().__init__()
        if out_dim is None: out_dim = d_model
        self.patch_size = 8 # 8 32
        enc_dim = dec_dim = d_model
        self.encoder = Encoder(self.patch_size, in_dim, enc_dim, out_dim=out_dim, n_heads=n_heads, nlayers=nlayers, drop=drop)
        self.decoder = Decoder(enc_dim, dec_dim, out_dim=self.patch_size*in_dim, n_heads=n_heads, nlayers=nlayers, drop=drop)

    def loss(self, x): # [b,t,in]
        b,t,_ = x.shape
        # print(x.shape)
        context_indices, trg_indices = random_masking(t//self.patch_size, .3, b=b)
        # print('x',x.shape, context_indices.shape, trg_indices.shape)
        sx = self.encoder(x, mask_indices=context_indices) # [b, num_context_toks, d]
        # print('seq_jepa loss sx',sx.shape)
        y_ = self.decoder(sx, mask_indices=context_indices, trg_indices=trg_indices) # [b,t,out]
        y = patchify(x, self.patch_size)
        # print(y.shape, y_.shape)
        # loss = F.mse_loss(y, y_)
        loss = F.mse_loss(y[torch.arange(b).unsqueeze(-1), trg_indices], y_[torch.arange(b).unsqueeze(-1), trg_indices])
        return loss

    def forward(self, x): # [b,t,d]
        return self.encoder(x).mean(dim=1)

try:
    in_dim = X[0].shape[-1] # 3
    out_dim = train_data.vocab_size # 16
except NameError:
    in_dim, out_dim = 3,16
print(in_dim, out_dim)
d_model=64
mae = MAE(in_dim=in_dim, d_model=d_model, out_dim=None, nlayers=1, n_heads=8, drop=.0).to(device)#.to(torch.float)
optim = torch.optim.AdamW(mae.parameters(), lr=1e-3) # 1e-3? default 1e-2
# optim = torch.optim.AdamW(mae.parameters(), lr=3e-4) # 1e-3? default 1e-2
# optim = torch.optim.AdamW([{'params': mae.encoder.parameters()},
#     {'params': mae.decoder.parameters(), 'lr': 3e-3}], lr=1e-3, weight_decay=1e-2) # default 1e-2, 5e-2
    # {'params': mae.decoder.parameters(), 'lr': 1e-2}], lr=1e-3, weight_decay=1e-2)


# https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml
# d_model 1024,384
# depth 12,6/12
# wd 5e-2 - 4e-1
# adamw 1e-4 - 1e-3 - 1e-6
# ema 0.996-1

print(sum(p.numel() for p in mae.parameters() if p.requires_grad)) # 27584

x = torch.rand((24, 1600, in_dim), device=device)
out = mae.loss(x)
print(out.shape)

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)
    def forward(self, x): return self.classifier(x)
classifier = Classifier(d_model, out_dim).to(device)
coptim = torch.optim.SGD(classifier.parameters(), lr=1e-3) # 1e-3
