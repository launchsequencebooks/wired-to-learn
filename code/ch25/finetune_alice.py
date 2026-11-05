# Adapting the Shakespeare mini-GPT of Chapter 18 to a new domain, Alice in Wonderland:
# training from scratch vs. full fine-tuning vs. LoRA. We also measure how much
# Shakespeare each model forgets.
import sys, json, time, copy
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "ch17")); sys.path.insert(0, str(root / "ch18"))
import torch
from torch import nn
from wtlkit import load_text, set_seed
from mini_gpt import MiniGPT, BLOCK
# (importing Chapter 17's data module prints its Shakespeare stats: the first line)
from char_lstm import chars, stoi, val_data as shakespeare_val
from lora import add_lora, LoRALinear


def extras(model):
    """The small modules trained alongside the LoRA paths (peft: modules_to_save)."""
    mods = [model.tok, model.pos, model.head]
    mods += [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
    return [p for m in mods for p in m.parameters()]


def lora_model(r=8):
    model = pretrained()
    for p in model.parameters():
        p.requires_grad = False                    # freeze EVERYTHING in the pretrained model ...
    add_lora(model, r=r, targets=("qkv", "proj", "0", "2"))   # ... then add LoRA paths
    for p in extras(model):
        p.requires_grad = True   # embeddings, norms and head stay trainable: cheap,
    return model                 # and they carry the new domain's character statistics

# ---- the new domain, mapped onto the 65-character Shakespeare vocabulary
raw = load_text("alice.txt")
table = {'"': "'", "(": ",", ")": ",", "[": "", "]": "", "*": "", "_": ""}
alice = "".join(table.get(c, c if c in stoi else " ") for c in raw)
data = torch.tensor([stoi[c] for c in alice])
n = int(0.9 * len(data))
alice_train, alice_val = data[:n], data[n:]
print(f"Alice: {len(alice):,} characters ({len(alice_train):,} train / {len(alice_val):,} validation)")


def batch(d, bs=32):
    ix = torch.randint(len(d) - BLOCK - 1, (bs,))
    return torch.stack([d[i:i + BLOCK] for i in ix]), torch.stack([d[i + 1:i + BLOCK + 1] for i in ix])


@torch.no_grad()
def loss_on(model, d, iters=30):
    model.eval()
    g = torch.Generator().manual_seed(123)                       # same batches for every model
    total = 0.0
    for _ in range(iters):
        ix = torch.randint(len(d) - BLOCK - 1, (32,), generator=g)
        x = torch.stack([d[i:i + BLOCK] for i in ix]); y = torch.stack([d[i + 1:i + BLOCK + 1] for i in ix])
        total += nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    return total / iters


def pretrained():
    path = root / "ch18" / "mini_gpt.pt"
    assert path.exists(), "run code/ch18/mini_gpt.py first to create mini_gpt.pt"
    m = MiniGPT(len(chars)); m.load_state_dict(torch.load(path))
    return m


def train(model, steps=600, lr=1e-3):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        x, y = batch(alice_train)
        loss = nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward(); opt.step()
    return model


base = pretrained()
print(f"pretrained model, before any Alice training: Alice val {loss_on(base, alice_val):.3f} | "
      f"Shakespeare val {loss_on(base, shakespeare_val):.3f}")

runs = {
    "from scratch":      (lambda: MiniGPT(len(chars)), 3e-3),
    "full fine-tuning":  (pretrained, 3e-4),
    "LoRA (r=8)":        (lora_model, 3e-3),
}
results = {}
print(f"\n{'method':18s}{'trainable':>11s}{'Alice val':>11s}{'Shakespeare val':>17s}{'time':>7s}")
for name, (make, lr) in runs.items():
    set_seed(0)
    model = make()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t0 = time.perf_counter()
    train(model, lr=lr)
    a, s = loss_on(model, alice_val), loss_on(model, shakespeare_val)
    results[name] = {"params": n_train, "alice": a, "shakespeare": s}
    print(f"{name:18s}{n_train:>11,}{a:>11.3f}{s:>17.3f}{time.perf_counter() - t0:>6.0f}s")
    if name == "LoRA (r=8)":
        lora_layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
        with torch.no_grad():                  # switch the adapter off: zero the LoRA
            tuned = [p.clone() for p in extras(model)]   # paths, and put the pretrained
            for p, q in zip(extras(model), extras(pretrained())):   # small modules back
                p.copy_(q)
        for m in lora_layers:
            m.scale_on, m.scale = m.scale, 0.0
        print(f"{'  adapter off':18s}{'':>11s}{loss_on(model, alice_val):>11.3f}"
              f"{loss_on(model, shakespeare_val):>17.3f}")
        with torch.no_grad():                      # ... and the whole adapter back on
            for p, t in zip(extras(model), tuned):
                p.copy_(t)
        for m in lora_layers:
            m.scale = m.scale_on
        torch.manual_seed(7)
        print("\nLoRA model writes:\n" + model.generate("Alice", 200, temperature=0.7))
json.dump(results, open("finetune_results.json", "w"))
