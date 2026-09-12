"""Chapter 30. Training the model, and watching it learn.

The architecture is chapter 29's with a batch axis added. Everything
else here is the training loop: batches, a loss, an optimizer, a
schedule, and a measurement that says whether the model learned the
thing it was supposed to learn.
"""
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ch30_corpus import build, is_grammatical

CONTEXT = 64
D_MODEL = 96
N_HEADS = 4
N_LAYERS = 3
D_FF = 384
BATCH = 32
STEPS = 1200
WARMUP = 100
PEAK_RATE = 3e-3
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "ch30_results.json")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, D_FF)
        self.down = nn.Linear(D_FF, D_MODEL)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D_MODEL, dim=-1)
        shape = (b, n, N_HEADS, D_MODEL // N_HEADS)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(a.transpose(1, 2).reshape(b, n, D_MODEL))
        return x + self.down(F.gelu(self.up(self.ln2(x)),
                                    approximate="tanh"))


class Model(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab, D_MODEL)
        self.pos = nn.Embedding(CONTEXT, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.final = nn.LayerNorm(D_MODEL)
        self.apply(self._scale)

    @staticmethod
    def _scale(module):
        """Small initial weights. Chapter 7 says why the scale matters;
        without this the first loss is twenty times chance."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx):
        n = idx.shape[1]
        x = self.embed(idx) + self.pos(torch.arange(n, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.final(x) @ self.embed.weight.T


def batches(data, rng, size=BATCH):
    start = torch.randint(len(data) - CONTEXT - 1, (size,),
                          generator=rng)
    x = torch.stack([data[i:i + CONTEXT] for i in start])
    y = torch.stack([data[i + 1:i + CONTEXT + 1] for i in start])
    return x, y


def loss_on(model, data, rng, rounds=8):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(rounds):
            x, y = batches(data, rng)
            logits = model(x)
            total += float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1)))
    model.train()
    return total / rounds


def rate_at(step):
    if step < WARMUP:
        return PEAK_RATE * step / WARMUP
    progress = (step - WARMUP) / (STEPS - WARMUP)
    return PEAK_RATE * 0.5 * (1 + math.cos(math.pi * progress))


def generate(model, encode, decode, prompt="the ", count=240, seed=0):
    rng = torch.Generator().manual_seed(seed)
    idx = torch.tensor([encode(prompt)])
    for _ in range(count):
        logits = model(idx[:, -CONTEXT:])[0, -1]
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=rng)
        idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
    return decode(idx[0].tolist())


def flatten(text):
    """One line, so it fits an output block."""
    return " / ".join(part.strip() for part in text.split("\n")
                      if part.strip())


def grammatical_share(text):
    lines = [line for line in text.split(".")[1:-1]]
    lines = [line.strip() + "." for line in lines if line.strip()]
    if not lines:
        return 0.0, 0
    return sum(is_grammatical(line) for line in lines) / len(lines), \
        len(lines)


def train():
    torch.manual_seed(0)
    text = build()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(chars[i] for i in ids)
    data = torch.tensor(encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = Model(len(chars))
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_RATE,
                            weight_decay=0.01)
    rng = torch.Generator().manual_seed(1)
    eval_rng = torch.Generator().manual_seed(2)

    history, samples = [], []
    started = time.perf_counter()
    for step in range(STEPS + 1):
        for group in opt.param_groups:
            group["lr"] = rate_at(min(step, STEPS - 1))
        if step % 100 == 0:
            record = {"step": step,
                      "train": loss_on(model, train_data, eval_rng),
                      "val": loss_on(model, val_data, eval_rng)}
            history.append(record)
        if step in (0, 300, STEPS):
            text_out = generate(model, encode, decode)
            share, count = grammatical_share(text_out)
            samples.append({"step": step, "text": flatten(text_out),
                            "share": share, "sentences": count})
        if step == STEPS:
            break
        x, y = batches(train_data, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return {"history": history, "samples": samples,
            "seconds": time.perf_counter() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
            "vocab": len(chars), "chars": len(text)}


def results():
    """Train once and keep the outcome, so the book builds quickly."""
    if os.path.exists(RESULTS):
        return json.load(open(RESULTS))
    out = train()
    json.dump(out, open(RESULTS, "w"), indent=1)
    return out


def main():
    out = results()
    print("THE MODEL AND THE DATA")
    print(f"  {out['parameters']:,} parameters, vocabulary of "
          f"{out['vocab']}")
    print(f"  {out['chars']:,} characters, context of {CONTEXT}")
    print(f"  {STEPS} steps in {out['seconds']:.0f} seconds on a laptop")

    print("\nTHE LOSS, EVERY HUNDRED STEPS")
    print("  step    training   held out   gap")
    for row in out["history"]:
        gap = row["val"] - row["train"]
        print(f"  {row['step']:<7} {row['train']:<10.4f} "
              f"{row['val']:<10.4f} {gap:+.4f}")
    print(f"  chance for {out['vocab']} characters would be "
          f"{math.log(out['vocab']):.4f}")

    print("\nWHAT IT WRITES, AT THREE POINTS")
    for sample in out["samples"]:
        if sample["sentences"]:
            share = 100 * sample["share"]
            count = sample["sentences"]
            verdict = f"{share:.0f}% of {count} sentences pass"
        else:
            verdict = "no complete sentences at all"
        print(f"  after {sample['step']} steps, {verdict}")
        for start in (0, 68):
            piece = sample["text"][start:start + 68]
            if piece:
                print(f"    {piece}")


if __name__ == "__main__":
    main()
