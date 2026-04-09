import math  # math 임포트 추가
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# [1] Feed Forward Network (FFNN)
class FFNN(nn.Module):
    def __init__(self, d, dropout=0.2, bias=False):
        super().__init__()
        self.c_fc = nn.Linear(d, 4 * d, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d, d, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# [2] Causal Self-Attention
class CausalSelfAttention(nn.Module):
    def __init__(self, d, H, T, bias, dropout): # 인자 개별 수신으로 수정
        super().__init__()
        assert d % H == 0
        self.c_attn = nn.Linear(d, 3 * d, bias=bias)
        self.c_proj = nn.Linear(d, d, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = H
        self.d_embd = d
        
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(T, T)).view(1, 1, T, T))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_embd, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

# [3] Transformer Block
class Block(nn.Module):
    def __init__(self, d, H, T, bias, dropout): # 인자 개별 수신으로 수정
        super().__init__()
        self.ln_1 = nn.LayerNorm(d) # nn.LayerNorm 사용
        self.attn = CausalSelfAttention(d, H, T, bias, dropout)
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = FFNN(d, dropout, bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# [4] GPT 모델 메인
class GPT(nn.Module):
    def __init__(self, d, H, T, V, layers, bias=False, dropout=0.2):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(V, d),
            wpe=nn.Embedding(T, d),
            drop=nn.Dropout(dropout),
            blocks=nn.ModuleList([Block(d, H, T, bias, dropout) for _ in range(layers)]),
            ln_f=nn.LayerNorm(d),
            head=nn.Linear(d, V, bias=bias),
        ))

    def forward(self, idx, targets=None):
        device = idx.device
        _, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.blocks:
            x = block(x)
        
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.transformer.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.transformer.head(x[:, [-1], :])
            loss = None

        return logits, loss

# --- 테스트 실행부 ---
tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "Hello, I am a language model"
inputs = tokenizer(text, return_tensors="pt")
idx = inputs['input_ids']

model = GPT(d=32, H=4, T=64, V=50257, layers=2) # H=3에서 4로 수정 (d=32가 3으로 안나눠짐)

model.eval()
with torch.no_grad():
    logits, _ = model(idx)

next_token_idx = torch.argmax(logits, dim=-1)
next_token = tokenizer.decode(next_token_idx[0])

print(f"입력 텍스트: {text}")
print(f"예측 단어: '{next_token}'")