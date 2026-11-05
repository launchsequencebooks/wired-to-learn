# A character-level language model: an LSTM learns to write like Shakespeare.
import sys, json, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from wtlkit import load_text, set_seed

text = load_text("shakespeare.txt")
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: torch.tensor([stoi[c] for c in s])
decode = lambda ids: "".join(chars[i] for i in ids)
data = encode(text)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"{len(text):,} characters, vocabulary of {len(chars)}")


def get_batch(split, batch_size=64, block=100):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block - 1, (batch_size,))
    x = torch.stack([d[i:i + block] for i in ix])
    y = torch.stack([d[i + 1:i + block + 1] for i in ix])      # targets: the next character
    return x, y


class CharLSTM(nn.Module):
    def __init__(self, vocab, emb=64, hidden=256, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb)                  # a learned vector per character
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True, dropout=0.1)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, idx, state=None):
        h, state = self.lstm(self.embed(idx), state)
        return self.out(h), state

    @torch.no_grad()
    def generate(self, prompt, n_chars, temperature=1.0):
        self.eval()
        idx, state = encode(prompt)[None], None
        logits, state = self(idx, state)
        out = list(prompt)
        for _ in range(n_chars):
            probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)                    # sample the next character
            out.append(chars[nxt.item()])
            logits, state = self(nxt[None], state)               # carry the memory forward
        return "".join(out)


@torch.no_grad()
def estimate_loss(model, iters=20):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        out[split] = sum(nn.functional.cross_entropy(model(x)[0].flatten(0, 1), y.flatten()).item()
                         for x, y in (get_batch(split) for _ in range(iters))) / iters
    model.train()
    return out


if __name__ == "__main__":
    set_seed(0)
    model = CharLSTM(len(chars))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    STEPS, log, t0 = 1500, [], time.perf_counter()
    for step in range(STEPS + 1):
        if step % 300 == 0:
            l = estimate_loss(model)
            log.append((step, l["train"], l["val"]))
            print(f"step {step:4d}  train {l['train']:.3f}  val {l['val']:.3f}  "
                  f"({l['val'] / math.log(2):.2f} bits/char)  {time.perf_counter() - t0:.0f}s")
        if step == STEPS:
            break
        x, y = get_batch("train")
        logits, _ = model(x)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)          # standard for RNNs
        opt.step()
    torch.save(model.state_dict(), "char_lstm.pt")
    json.dump(log, open("char_lstm_log.json", "w"))
    torch.manual_seed(42)
    for temp in [0.5, 1.0]:
        print(f"\n--- temperature {temp} ---")
        print(model.generate("ROMEO:", 250, temperature=temp))
