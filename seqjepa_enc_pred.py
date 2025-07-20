# @title TransformerModel/Predictor
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TransformerPredictor(nn.Module):
    def __init__(self, in_dim, d_model, out_dim=None, n_heads=4, d_hid=None, nlayers=1, drop=0.):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)# if in_dim != d_model else None
        # self.pos_enc = RotEmb(d_model, top=1, base=10000)
        # self.pos_emb = nn.Parameter(torch.randn(1, 256, d_model)*.02) # 200
        self.pos_emb = RoPE(d_model, seq_len=7733, base=10000) # 256
        self.transformer = nn.Sequential(*[AttentionBlock(d_model, n_heads=n_heads, drop=drop) for _ in range(nlayers)])
        self.cls = nn.Parameter(torch.randn(1,1,d_model)*0.02) # randn zeros
        out_dim = out_dim or d_model
        self.norm = nn.RMSNorm(d_model) # LayerNorm RMSNorm
        self.lin = nn.Linear(d_model, out_dim)# if out_dim != d_model else None
        # torch.nn.init.normal_(self.embed.weight, std=.02)
        # if self.lin: torch.nn.init.normal_(self.lin.weight, std=.02)

    def forward(self, x, context_indices, trg_indices): # [batch, seq_len, d_model], [batch, seq_len] # True will be ignored by the attention # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        x = self.embed(x) # [batch, seq_len, d_model] or [batch, num_context_toks, d_model]
        batch, seq, dim = x.shape
        # print(max(context_indices))
        # print(context_indices.max())
        # print(context_indices)
        # if max(context_indices+trg_indices) > self.pos_emb.shape[1]: self.pos_emb = RoPE(self.d_model, seq_len=max(context_indices+trg_indices), base=10000) # 256
        # if max(context_indices.max()+trg_indices.max()) > self.pos_emb.shape[1]: self.pos_emb = RoPE(self.d_model, seq_len=max(context_indices.max()+trg_indices.max()), base=10000) # 256
        # if max(max(context_indices)+max(trg_indices)) > self.pos_emb.shape[1]: self.pos_emb = RoPE(self.d_model, seq_len=max(max(context_indices)+max(trg_indices)), base=10000) # 256
        # x = x * self.pos_enc(context_indices)
        # print("Trans pred",x.shape, self.pos_emb[0,context_indices].shape)
        # print("Trans pred",x.shape, self.pos_emb.shape)
        x = x + self.pos_emb[0,context_indices]
        # x = x * self.pos_emb[0,context_indices]
        # print('pred fwd', self.pos_emb[:,context_indices].shape)

        # pred_tokens = self.cls * self.pos_enc(trg_indices) # [M, num_trg_toks, d_model]
        pred_tokens = self.cls + self.pos_emb[0,trg_indices]
        # pred_tokens = self.cls * self.pos_emb[0,trg_indices]
        # print("pred fwd", x.shape, pred_tokens.shape)
        x = torch.cat([x, pred_tokens], dim=1) # [batch, seq_len+num_trg_toks, d_model]
        out = self.transformer(x)

        out = self.norm(out)
        out = out[:,seq:] # [batch, num_trg_toks, d_model]
        out = self.lin(out)
        return out # [seq_len, batch_size, ntoken]

# class SLSTM(nn.Module):
#     def __init__(self, d_model, num_layers=2, batch_first=True):
#         super().__init__()
#         self.lstm = nn.LSTM(d_model, d_model, num_layers)

#     def forward(self, x): # [b,c,t]
#         x = x + self.lstm(x.transpose(-2,-1))[0].transpose(-2,-1) # skip=True
#         return x


class TransformerModel(nn.Module):
    # def __init__(self, in_dim, d_model, out_dim=None, nhead=8, d_hid=None, nlayers=1, drop=0):
    def __init__(self, patch_size, in_dim, d_model, out_dim=None, n_heads=4, nlayers=1, drop=0):
        super().__init__()
        self.d_model = d_model
        patch_size=8
        act = nn.ReLU() # ReLU SiLU GELU
        # act = LearntSwwish(d_model) # SnakeBeta LearntSwwish
        # act1 = LearntSwwish(d_model) # SnakeBeta LearntSwwish
        # act = Swwish()
        self.embed = nn.Sequential(
            # # nn.Conv1d(in_dim, d_model,7,2,7//2), nn.MaxPool1d(2,2), #nn.MaxPool1d(3, 2, 3//2),
            nn.Conv1d(in_dim, d_model,7,2,7//2), nn.BatchNorm1d(d_model), act,
            nn.Conv1d(d_model, d_model,5,2,5//2), nn.BatchNorm1d(d_model), act,
            # nn.Conv1d(in_dim, d_model,3,2,3//2), nn.BatchNorm1d(d_model), act, nn.MaxPool1d(2,2),
            # nn.Conv1d(d_model, d_model,3,2,3//2), nn.BatchNorm1d(d_model), act, nn.MaxPool1d(2,2),
            nn.Conv1d(d_model, d_model,3,2,3//2),
            # nn.Conv1d(in_dim, d_model, patch_size, patch_size), # like patch
            # nn.Conv1d(in_dim, d_model, 1, 1), # like patch

            # nn.Conv2d(d_model, d_model,(in_dim,3),2,3//2),
            # nn.Conv1d(in_dim, d_model,7,2,7//2), nn.Dropout(drop), nn.BatchNorm1d(d_model), snake,
            # SLSTM(d_model),

            )
        # self.pos_enc = RotEmb(d_model, top=1, base=10000)
        # self.pos_emb = nn.Parameter(torch.randn(1, 256, d_model)*.02) # 200
        self.pos_emb = RoPE(d_model, seq_len=7733, base=10000) # 256
        # self.transformer = nn.Sequential(*[AttentionBlock(d_model, n_heads=n_heads) for _ in range(nlayers)])
        self.transformer = nn.Sequential(*[AttentionBlock(d_model, n_heads=n_heads, drop=drop) for _ in range(nlayers)])
        self.norm = nn.RMSNorm(d_model) # LayerNorm RMSNorm
        self.lin = nn.Linear(d_model, out_dim) if out_dim and out_dim != d_model else None
        # if self.lin: torch.nn.init.normal_(self.lin.weight, std=.02)

        # self.embed.apply(self.init_conv)
        # self.embed.apply(self.init_weights)

    # def init_conv(self, m):
    #     if isinstance(m, nn.Conv1d):
    #         # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         if m.bias is not None:
    #             # bound = 1 / math.sqrt(m.in_channels * m.kernel_size * m.kernel_size)
    #             # nn.init.uniform_(m.bias, -bound, bound)
    #             nn.init.zeros_(m.bias)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.zeros_(m.bias)


    def forward(self, x, context_indices=None): # [batch, num_context_toks, 3], [batch, num_context_toks] # True will be ignored by the attention # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        x = self.embed(x.transpose(-2,-1)).transpose(-2,-1) # [batch, T, d_model]
        # try: print("Trans fwd",x.shape, context_indices.shape)
        # except: print("Trans fwd noind",x.shape)
        # x = self.pos_enc(x)
        # if x.shape[1] > self.pos_emb.shape[1]: self.pos_emb = RoPE(self.d_model, seq_len=x.shape[1], base=10000) # 256
        x = x + self.pos_emb[:,:x.shape[1]]
        # x = x * self.pos_emb[:,:x.shape[1]]
        if context_indices != None: x = x[torch.arange(x.shape[0]).unsqueeze(-1), context_indices] # [batch, num_context_toks, d_model]

        # print("TransformerModel",x.shape)
        x = self.transformer(x)
        out = self.norm(x)
        if self.lin: out = self.lin(out)
        return out

batch, seq_len, d_model = 4,1751,16 # wisdm 3500, ethol conc 1751
in_dim = 3
patch_size=32
model = TransformerModel(patch_size, in_dim, d_model, n_heads=4, nlayers=3, drop=0.).to(device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad)) # 27584
x =  torch.rand((batch, seq_len, in_dim), device=device)
out = model(x)
print(out.shape)
# # # print(out)
# model = TransformerPredictor(in_dim, d_model, out_dim=None, d_head=4, d_hid=None, nlayers=1).to(device)
# out = model(out)
# print(out.shape)
# for name, param in model.named_parameters():
#     print(name, param.shape, param[0])
