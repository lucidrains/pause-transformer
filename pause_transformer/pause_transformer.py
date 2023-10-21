import torch
import torch.nn.functional as F
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList, Sequential

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# functions

def exists(v):
    return v is not None

# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# feedforward

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

class CausalAttention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim ** -0.5
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        )

        self.to_out = Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = x.device).triu(j - i + 1)

        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        return self.to_out(out)

# integrate previous pause / thinking information

class IntegratePreviousThought(Module):
    def __init__(self, dim):
        super().__init__()
        self.net =  Sequential(
            RMSNorm(dim),
            Rearrange('b n p d -> b n (p d)'),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, pause_tokens):
        p = pause_tokens[:, :, -1]
        p = F.pad(p, (0, 0, 1, -1), value = 0.)

        x = torch.stack((x, p), dim = -2)
        out = self.net(x)
        return out

# class

class PauseTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_pause_length = 2,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.max_pause_length = max_pause_length

        self.pause_tokens = nn.Parameter(torch.randn(max_pause_length, dim))

        self.integrate_prev_pause = IntegratePreviousThought(dim)

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.to_logits = Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

    def forward(
        self,
        x,
        return_loss = False,
        no_prev_pause_integration = False
    ):
        """
        einstein notation:
        b - batch
        n - main sequence length
        p - thinking sequence length (pause)
        d - feature dimension
        """

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        batch, seq_len = x.shape

        x = self.token_emb(x)

        p = repeat(self.pause_tokens, 'p d -> b n p d', b = batch, n = seq_len)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

            # now process the thinking tokens

            x, ps = pack([x, p], 'b n * d')
            x = rearrange(x, '... p d -> (...) p d')

            x = attn(x) + x
            x = ff(x) + x

            x = rearrange(x, '(b n) p d -> b n p d', b = batch)
            x, p = unpack(x, ps, 'b n * d')

            # during training, should allow for forcing each token to think independent of previous token's thinking

            if no_prev_pause_integration:
                continue

            # integrating the previous pause token

            x = x + self.integrate_prev_pause(x, p)

        x, _ = pack([x, p], 'b n * d')

        logits = self.to_logits(x)

        if not return_loss:
            return logits

        labels = repeat(labels, 'b n -> (b p) n', p = self.max_pause_length + 1)

        logits = rearrange(logits, 'b n p d -> (b p) d n')

        loss = F.cross_entropy(logits, labels)
        return loss
