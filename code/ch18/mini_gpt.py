# A miniature GPT: a decoder-only Transformer trained on Shakespeare with the same
# data, steps and batch size as Chapter 17's LSTM.
import sys, json, time, math
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "ch17"))
import torch
from torch import nn
from wtlkit import set_seed
from char_lstm import chars, encode, decode, get_batch     # same data and split as Chapter 17

BLOCK = 100                                                # context length (characters)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)         # all three projections at once
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK, BLOCK)).bool())
        self.last_weights = None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, dk): each head attends independently
        q, k, v = (t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))   # no peeking ahead
        att = torch.softmax(att, dim=-1)
        self.last_weights = att.detach()
        y = (att @ v).transpose(1, 2).reshape(B, T, C)     # concatenate the heads
        return self.proj(y)


class Block(nn.Module):
    """Pre-norm Transformer block: x + Attention(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))                     # tokens exchange information
        x = x + self.mlp(self.ln2(x))                      # each token thinks on its own
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)            # what the character is
        self.pos = nn.Embedding(BLOCK, d_model)            # where it is (learned positions)
        self.blocks = nn.Sequential(*[Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        return self.head(self.ln(self.blocks(x)))

    @torch.no_grad()
    def generate(self, prompt, n_chars, temperature=1.0):
        self.eval()
        idx = encode(prompt)[None]
        for _ in range(n_chars):
            logits = self(idx[:, -BLOCK:])[0, -1] / temperature   # only the last BLOCK chars fit
            nxt = torch.multinomial(torch.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt[None]], dim=1)
        return decode(idx[0].tolist())


@torch.no_grad()
def estimate_loss(model, iters=20):
    model.eval()
    out = {s: sum(nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
                  for x, y in (get_batch(s) for _ in range(iters))) / iters for s in ["train", "val"]}
    model.train()
    return out


if __name__ == "__main__":
    set_seed(0)
    model = MiniGPT(len(chars))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    STEPS, WARMUP = 1500, 100
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(           # warmup + cosine (Chapter 11)
        opt, lambda s: (s + 1) / WARMUP if s < WARMUP else
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / (STEPS - WARMUP))))
    log, t0 = [], time.perf_counter()
    for step in range(STEPS + 1):
        if step % 300 == 0:
            l = estimate_loss(model)
            log.append((step, l["train"], l["val"]))
            print(f"step {step:4d}  train {l['train']:.3f}  val {l['val']:.3f}  "
                  f"({l['val'] / math.log(2):.2f} bits/char)  {time.perf_counter() - t0:.0f}s")
        if step == STEPS:
            break
        x, y = get_batch("train")
        loss = nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    torch.save(model.state_dict(), "mini_gpt.pt")
    json.dump(log, open("mini_gpt_log.json", "w"))
    torch.manual_seed(42)
    for temp in [0.5, 1.0]:
        print(f"\n--- temperature {temp} ---")
        print(model.generate("ROMEO:", 250, temperature=temp))
