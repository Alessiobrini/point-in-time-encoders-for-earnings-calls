"""The ChronoGPT decoder, vendored for scoring (D32).

Copied from `ChronoGPT_instruct.py` in `manelalab/chrono-gpt-instruct-v1-<Y>1231` (MIT license, He, Lv,
Manela and Wu), with three changes that leave the computed logits unchanged:

- `Rotary` builds its tables for 2,048 positions instead of 65,536. The angles of a position do not
  depend on the table length, and the model's context is 1,792 tokens, so the first 2,048 rows are the
  same numbers. The full tables cost 3.5 GB across 52 layers on an 11 GB card.
- The generation helpers and the Hub mixin are dropped, since D32 only scores fixed answers.
- `load` reads the checkpoint from a local Hugging Face cache and casts the weights to bfloat16, the
  dtype the forward pass already casts its activations to.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT = 1792
EOS_ID = 50256
SYSTEM_PROMPT = """You are ChronoGPT, a large language model trained by ManelaLab at WashU.
    Below is an instruction that describes a task.
    Write a response that appropriately completes the request."""


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def template(input_text: str) -> str:
    """The instruction format of `extract_response` in the model's own code, up to the response."""
    return f"\n\n### Instruction:\n{SYSTEM_PROMPT}\n{input_text}\n\n### Input:\n### Response:\n"


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x):
        cos, sin = self.cos[None, :x.size(-3), None, :], self.sin[None, :x.size(-3), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(self.head_dim)
        self.c_proj = CastedLinear(dim, dim)

    def forward(self, x, ve):
        B, T = x.size(0), x.size(1)
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.attn = CausalSelfAttention(model_dim, num_heads)
        self.mlp = MLP(model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, ve, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(norm(x), ve)
        return x + self.mlp(norm(x))


class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers=52):
        super().__init__()
        self.num_layers = num_layers
        self.embed = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])

    def forward(self, inputs):
        base = [emb(inputs).bfloat16() for emb in self.embed]
        half = self.num_layers // 2
        encoder = [base[i] if i < 3 else None for i in range(half)]
        decoder = [base[i - (half - 3)] if i >= (half - 3) else None for i in range(half)]
        return encoder + decoder


class ChronoGPT(nn.Module):
    def __init__(self, vocab_size, num_layers, num_heads, model_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads) for _ in range(num_layers)])
        self.value_embeds = ValueEmbedding(vocab_size, model_dim, num_layers=num_layers)
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

    @torch.inference_mode()
    def forward(self, inputs):
        """Logits for every position, float32, shape (B, T, vocab)."""
        x0 = norm(self.embed(inputs).bfloat16())
        x = x0
        # the value embeddings of a batch, layer by layer (the original builds them per sequence)
        ve = self.value_embeds(inputs)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0)
        logits = self.lm_head(norm(x))
        return (15 * torch.tanh(logits / 15)).float()


def load(repo_id: str, cache_dir: str, device: str = "cuda") -> ChronoGPT:
    """A ChronoGPT checkpoint from a local cache, in bfloat16 on `device`, in eval mode."""
    from huggingface_hub import hf_hub_download
    cfg = torch.load(hf_hub_download(repo_id, "config.pt", cache_dir=cache_dir), weights_only=False)
    model = ChronoGPT(**cfg)
    state = torch.load(hf_hub_download(repo_id, "pytorch_model.bin", cache_dir=cache_dir),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    del state
    return model.to(dtype=torch.bfloat16).to(device).eval()
