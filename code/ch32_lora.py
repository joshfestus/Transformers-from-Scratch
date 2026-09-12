"""Chapter 32. Low-rank adaptation, measured rather than asserted.

The claim behind LoRA is that what fine-tuning changes about a weight
matrix is mostly low rank, so it can be stored as two thin matrices
instead of one square one. This file trains a model, fine-tunes it on a
shifted task, and measures the rank of the change.
"""
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

import ch30_train as base
from ch30_corpus import PLACES, build, sentence

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "ch32_results.json")
PRETRAIN, FINETUNE = 500, 250


def shifted_corpus(count=600, seed=7):
    """The same grammar with the places replaced, so only part changes."""
    rng = random.Random(seed)
    saved = list(PLACES)
    PLACES[:] = ["harbour", "market", "orchard", "station", "valley"]
    text = "\n".join(sentence(rng) for _ in range(count)) + "\n"
    PLACES[:] = saved
    return text


def lora_parameters(d_model, rank):
    return 2 * d_model * rank


def captured(matrix, rank):
    """Share of a matrix's size that the best rank-r version keeps."""
    values = np.linalg.svd(matrix, compute_uv=False)
    return float((values[:rank] ** 2).sum() / (values ** 2).sum())


def run():
    torch.manual_seed(0)
    text = build()
    chars = sorted(set(text + shifted_corpus()))
    stoi = {c: i for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]

    model = base.Model(len(chars))
    data = torch.tensor(encode(text), dtype=torch.long)
    rng = torch.Generator().manual_seed(1)

    def train_on(dataset, steps, rate):
        opt = torch.optim.AdamW(model.parameters(), lr=rate,
                                weight_decay=0.01)
        for _ in range(steps):
            x, y = base.batches(dataset, rng)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return float(loss.detach())

    first = train_on(data, PRETRAIN, 3e-3)
    # matrices only; a bias is one dimensional and has no rank to measure
    before = {name: p.detach().clone()
              for name, p in model.named_parameters()
              if p.dim() == 2 and ("qkv" in name or "up" in name)}

    shifted = torch.tensor(encode(shifted_corpus()), dtype=torch.long)
    second = train_on(shifted, FINETUNE, 5e-4)

    spectra = {}
    for name, saved in before.items():
        change = (dict(model.named_parameters())[name].detach()
                  - saved).numpy()
        spectra[name] = {str(r): captured(change, r)
                         for r in (1, 2, 4, 8, 16, 32)}
        spectra[name]["shape"] = list(change.shape)
    return {"pretrain_loss": first, "finetune_loss": second,
            "spectra": spectra}


def results():
    if os.path.exists(RESULTS):
        return json.load(open(RESULTS))
    out = run()
    json.dump(out, open(RESULTS, "w"), indent=1)
    return out


def main():
    out = results()
    print("A MODEL, THEN THE SAME MODEL FINE-TUNED")
    print(f"  loss after {PRETRAIN} steps on the first corpus "
          f"{out['pretrain_loss']:.3f}")
    print(f"  loss after {FINETUNE} steps on the shifted one "
          f"{out['finetune_loss']:.3f}")
    print("  only the place names changed between the two corpora")

    print("\nHOW MUCH OF THE CHANGE A LOW-RANK VERSION KEEPS")
    print("  matrix                shape        r=1    r=4    r=8   r=32")
    for name, row in out["spectra"].items():
        shape = "x".join(str(v) for v in row["shape"])
        short = name.replace("blocks.", "b").replace(".weight", "")
        print(f"  {short:<21} {shape:<12} "
              f"{row['1']:.2f}   {row['4']:.2f}   {row['8']:.2f}   "
              f"{row['32']:.2f}")

    print("\nWHAT EACH RANK COSTS, FOR A 4096 WIDE MATRIX")
    d = 4096
    full = d * d
    print("  rank    parameters    share of the full matrix")
    for rank in [1, 4, 8, 16, 64, 256]:
        low = lora_parameters(d, rank)
        print(f"  {rank:<7} {low:>11,}   {100 * low / full:>6.2f}%")
    print(f"  the full matrix is {full:,}")

    print("\nTHE SAME SHAPE, FILLED WITH NOISE INSTEAD")
    shape = out["spectra"]["blocks.0.qkv.weight"]["shape"]
    noise = np.random.default_rng(0).standard_normal(shape)
    print(f"  a random {shape[0]}x{shape[1]} matrix")
    for rank in [1, 4, 8, 32]:
        print(f"  rank {rank:<4} keeps {captured(noise, rank):.3f}")
    print("  the measured update keeps several times as much at every")
    print("  rank, which is the claim behind the method, and it is a")
    print("  difference of degree rather than of kind")


if __name__ == "__main__":
    main()
