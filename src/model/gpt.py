from src.model.attention import MultHeadAttention

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self,emb_dim:int):
        super().__init__()

        self.eps = 1e-5
        self.scale = torch.tensor(torch.ones(emb_dim))
        self.shift = torch.tensor(torch.zeros(emb_dim))

    def forward(self,x:torch.Tensor):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True,unbiased=False)

        normed = (x-mean)/torch.sqrt(var+self.eps)

        return self.scale * normed + self.shift


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device, dtype=x.dtype)) *
            (x + 0.044715 * x.pow(3))
        ))

class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
            GeLU(),
            nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"])
        )

    def forward(self,x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attn = MultHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout_shortcut = nn.Dropout(cfg["drop_rate"])

    
    def forward(self,x):
        #For attn block
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout_shortcut(x)
        x = x+shortcut

        #For ff network
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout_shortcut(x)
        x = x+shortcut

        return x

class   GPT2Model(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"],cfg["emb_dim"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)
        self.dropout = nn.Dropout(cfg["drop_rate"])
    
    def forward(self,in_idx):
        batch_size,seq_len = in_idx.shape
        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        x = tok_emb + pos_emb
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

