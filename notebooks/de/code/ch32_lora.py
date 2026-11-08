"""Kapitel 32. Low-Rank Adaptation, gemessen statt behauptet.

Die Behauptung hinter LoRA ist, dass das, was Fine-Tuning an einer
Gewichtsmatrix ändert, überwiegend niedrigen Rang hat und sich daher als
zwei schmale Matrizen statt einer quadratischen speichern lässt. Diese
Datei trainiert ein Modell, passt es auf eine verschobene Aufgabe an und
misst den Rang der Änderung.
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
    """Dieselbe Grammatik mit anderen Orten, also ändert sich nur ein
    Teil."""
    rng = random.Random(seed)
    saved = list(PLACES)
    PLACES[:] = ["hafen", "markt", "bahnhof", "tal", "schule"]
    text = "\n".join(sentence(rng) for _ in range(count)) + "\n"
    PLACES[:] = saved
    return text


def lora_parameters(d_model, rank):
    return 2 * d_model * rank


def captured(matrix, rank):
    """Anteil der Größe einer Matrix, den die beste Version mit Rang r
    behält."""
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
    # nur Matrizen, ein Bias ist eindimensional und hat keinen Rang
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
    print("EIN MODELL, DANN DASSELBE MODELL NACH DEM FINE-TUNING")
    print(f"  Loss nach {PRETRAIN} Schritten auf dem ersten Korpus "
          f"{out['pretrain_loss']:.3f}")
    print(f"  Loss nach {FINETUNE} Schritten auf dem verschobenen "
          f"{out['finetune_loss']:.3f}")
    print("  zwischen den Korpora haben sich nur die Orte geändert")

    print("\nWIE VIEL DER ÄNDERUNG EIN NIEDRIGER RANG BEHÄLT")
    print("  Matrix                Form         r=1    r=4    r=8   r=32")
    for name, row in out["spectra"].items():
        shape = "x".join(str(v) for v in row["shape"])
        short = name.replace("blocks.", "b").replace(".weight", "")
        print(f"  {short:<21} {shape:<12} "
              f"{row['1']:.2f}   {row['4']:.2f}   {row['8']:.2f}   "
              f"{row['32']:.2f}")

    print("\nWAS JEDER RANG KOSTET, BEI EINER MATRIX DER BREITE 4096")
    d = 4096
    full = d * d
    print("  Rang    Parameter     Anteil an der vollen Matrix")
    for rank in [1, 4, 8, 16, 64, 256]:
        low = lora_parameters(d, rank)
        print(f"  {rank:<7} {low:>11_}   {100 * low / full:>6.2f} %")
    print(f"  die volle Matrix hat {full:_}")

    print("\nDIESELBE FORM, STATTDESSEN MIT RAUSCHEN GEFÜLLT")
    shape = out["spectra"]["blocks.0.qkv.weight"]["shape"]
    noise = np.random.default_rng(0).standard_normal(shape)
    print(f"  eine zufällige Matrix der Form {shape[0]}x{shape[1]}")
    for rank in [1, 4, 8, 32]:
        print(f"  Rang {rank:<4} behält {captured(noise, rank):.3f}")
    print("  die gemessene Änderung behält bei jedem Rang mehr,")
    print("  und das ist die Behauptung hinter der Methode, ein")
    print("  Unterschied im Grad und nicht in der Art")


if __name__ == "__main__":
    main()
