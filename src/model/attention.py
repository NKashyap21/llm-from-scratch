import torch
import torch.nn as nn 

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in,d_out):
        super().__init__()
        self.d_out = d_out
        self.W_q = nn.Parameter(torch.rand(d_in,d_out),requires_grad=True)
        self.W_k = nn.Parameter(torch.rand(d_in,d_out),requires_grad=True)
        self.W_v = nn.Parameter(torch.rand(d_in,d_out),requires_grad=True)

    def forward(self,x):
        queries = x @ self.W_q  #(n,d_in) @ (d_in,d_out)
        keys = x @ self.W_k
        values = x @ self.W_v

        #Caluculating attn_weights
        attn_scores = queries @ keys.T #(n,d_out) @ (d_out,n) = (n,n)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5  ,dim=-1
            )
        
        context_vec = attn_weights @ values #(n,d_out) 
        return context_vec 

class SelfAttention_v2(nn.Module):
    def __init__(self,d_in,d_out,qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_q = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_k = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_v = nn.Linear(d_in,d_out,bias=qkv_bias)

    def forward(self,x):
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        #Calculating Attn weights
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim = -1
        )

        #Context vector
        context_vec = attn_weights @ values
        return context_vec

class CasualAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_q = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_k = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_v = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length),diagonal=1)
        )

    def forward(self,x):
        b,num_tokens,d_in = x.shape

        queries = self.W_q(x) #(b,num_tokens,d_out)
        keys = self.W_k(x) #(b,num_tokens,d_out)
        values = self.W_v(x) #(b,num_tokens,d_out)

        #Calculating Attn_weights 
        attn_scores = queries @ keys.transpose(1,2) #(b,num_tokens,num_tokens)
        attn_scores = torch.masked_fill(
            input=attn_scores,
            mask = self.mask.bool()[:num_tokens,:num_tokens],
            value= -torch.inf
        ) 
        attn_weights = torch.softmax(
            input = attn_scores / keys.shape[-1]**0.5,
            dim = -1,
        ) #(b,num_tokens,num_tokens)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values #(b,num_tokens,d_out)
        return context_vec

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self,d_in,d_out,num_heads,context_length,dropout,qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CasualAttention(d_in,d_out,context_length,dropout,qkv_bias) for _ in range(num_heads)])

    def forward(self,x):
        return torch.cat([head(x) for head in self.heads],dim=-1)

class MultHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,num_heads,context_length,dropout,qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out should be divisble by num_heads" 

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_q = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_k = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_v = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length),diagonal=1),
        )
        self.out_proj = nn.Linear(d_out,d_out)

    def forward(self,x):
        b,num_tokens,d_in = x.shape
        
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        #Splitting the attention heads
        keys = keys.view(b,num_tokens,self.num_heads,self.head_dim) 
        values = values.view(b,num_tokens,self.num_heads,self.head_dim)
        queries = queries.view(b,num_tokens,self.num_heads,self.head_dim)

        keys = torch.transpose(keys,1,2) # (n,num_heads,num_tokens,head_dim)
        values = torch.transpose(values,1,2) # (n,num_heads,num_tokens,head_dim)
        queries = torch.transpose(queries,1,2) # (n,num_heads,num_tokens,head_dim)

        #Calculating Attention Weights
        attn_scores = queries @ keys.transpose(2,3) # (n,num_heads,num_tokens,head_dim) @  (n,num_heads,num_tokens,head_dim)
        attn_scores = torch.masked_fill(
            input=attn_scores,
            mask = self.mask.bool()[:num_tokens,:num_tokens],
            value=-torch.inf, 
        )

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim=-1,
        )

        attn_weights = self.dropout(attn_weights)

        #Calculating context_vec
        context_vec = (attn_weights @ values).transpose(1,2) 
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
