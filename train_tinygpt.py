import torch
import torch.nn as nn
from dataclasses import dataclass
import math
from torch.nn import functional as F

# --------------------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # n_embd is divisible by n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # q, k, v projection in a batch
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        qkv = self.c_attn(x)
        B, T, C = x.size()
        q, k, v = qkv.split(C, dim=2) # each with shape (B, T, C)
        # multi-headed attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # why this critical?
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (b, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # reassemble all heads
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()        
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config # is this really needed?

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, idx):
        # idx is of size (B, T)
        B, T = idx.size()
        assert T <=self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape of (T)
        pos_emb = self.transformer.wpe(pos) # (T, C); input is a vector representing positions
        tok_emb = self.transformer.wte(idx) # (B, T, C); input token sequence
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size); output is logits
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),    # 124M
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),   # 358M
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),   # 774M
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M    
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffering; attn is the attention block

        # init a huggingface transformer model
        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the params are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys) == len(sd_keys_hf), f'mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}'
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t()) # need to transpose
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k]) # vanilla copy
        
        return model

# ------------------------------------------------------------------------------
num_return_sequences = 5
max_length = 50

model = GPT.from_pretrained('gpt2')
model.eval() # good practice to put model to eval mode during generation
model.to('cuda')

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to('cuda')

# generate
# set seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length: # generate when not excedding max_length
    # forward the model to get logits
    with torch.no_grad(): # put to no_grad when generating
        logits = model(x)
        logits = logits[:, -1, :] # (B, vocab_size); only take the logits of last sequence position
        probs = F.softmax(logits, dim=-1)
        # top-k sampling
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # both (5, 50)
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1); draw 1 from topk_probs
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)