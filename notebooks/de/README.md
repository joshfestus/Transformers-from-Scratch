# Transformer von Grund auf: die deutschen Notebooks

Die Notebooks zu *Transformer von Grund auf: Das neuronale Netz hinter großen
Sprachmodellen (LLM) für Einsteiger*, von Josh Festus, der deutschen Ausgabe
von *Transformers from Scratch*.

Jedes Notebook öffnet sich in Colab und braucht keine Installation. Klick auf
ein Abzeichen unten. Die erste Zelle holt die zwei oder drei Dateien, die das
Kapitel braucht, aus dem Ordner `code` neben diesen Notebooks, und die übrigen
Zellen laufen der Reihe nach.

Der Code hier ist derselbe wie im Buch, mit deutschen Kommentaren und
Ausgaben. Er rechnet mit denselben Zahlen wie die englischen Originale im
Ordner `code` ganz oben im Repository. Die Beispiele sind deutsch: „die Katze
trank Milch“ statt „the cat drank milk“, und das Modell aus Kapitel 30 lernt
eine kleine deutsche Grammatik.

Nichts hier braucht eine Grafikkarte. Das Längste ist das Training in
Kapitel 30, und das dauert auf einem Laptop etwa anderthalb Minuten.

## Die Kapitel

| Kapitel | Notebook | |
|---|---|---|
| 1 | [Was ein neuronales Netz wirklich ist](01_was_ein_neuronales_netz_wirklich_ist.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/01_was_ein_neuronales_netz_wirklich_ist.ipynb) |
| 2 | [Vektoren, Matrizen und die Sprache der Formen](02_vektoren_matrizen_und_die_sprache_der_formen.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/02_vektoren_matrizen_und_die_sprache_der_formen.ipynb) |
| 3 | [Matrixmultiplikation als Informationsfluss](03_matrixmultiplikation_als_informationsfluss.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/03_matrixmultiplikation_als_informationsfluss.ipynb) |
| 4 | [Warum gestapelte lineare Schichten nichts bringen](04_warum_gestapelte_lineare_schichten_nichts_bringen.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/04_warum_gestapelte_lineare_schichten_nichts_bringen.ipynb) |
| 5 | [Die Softmax, oder wie aus Scores Wahrscheinlichkeiten werden](05_die_softmax_oder_wie_aus_scores_wahrscheinlichkeiten_werden.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/05_die_softmax_oder_wie_aus_scores_wahrscheinlichkeiten_werden.ipynb) |
| 6 | [Wie ein Netz lernt](06_wie_ein_netz_lernt.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/06_wie_ein_netz_lernt.ipynb) |
| 7 | [Gradientenabstieg, Lernraten und Optimierer](07_gradientenabstieg_lernraten_und_optimierer.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/07_gradientenabstieg_lernraten_und_optimierer.ipynb) |
| 8 | [Tokenisierung](08_tokenisierung.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/08_tokenisierung.ipynb) |
| 9 | [Embeddings](09_embeddings.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/09_embeddings.ipynb) |
| 10 | [Die Position, die Information, die Attention wegwirft](10_die_position_die_information_die_attention_wegwirft.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/10_die_position_die_information_die_attention_wegwirft.ipynb) |
| 11 | [Das Problem, für das die Attention erfunden wurde](11_das_problem_fuer_das_die_attention_erfunden_wurde.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/11_das_problem_fuer_das_die_attention_erfunden_wurde.ipynb) |
| 12 | [Query, Key und Value](12_query_key_und_value.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/12_query_key_und_value.ipynb) |
| 13 | [Scaled Dot-Product Attention, Symbol für Symbol](13_scaled_dot_product_attention_symbol_fuer_symbol.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/13_scaled_dot_product_attention_symbol_fuer_symbol.ipynb) |
| 14 | [Eine komplette Rechnung mit echten Zahlen](14_eine_komplette_rechnung_mit_echten_zahlen.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/14_eine_komplette_rechnung_mit_echten_zahlen.ipynb) |
| 15 | [Multi-Head Attention](15_multi_head_attention.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/15_multi_head_attention.ipynb) |
| 16 | [Masken](16_masken.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/16_masken.ipynb) |
| 17 | [Cross-Attention](17_cross_attention.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/17_cross_attention.ipynb) |
| 18 | [Die Attention schnell machen](18_die_attention_schnell_machen.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/18_die_attention_schnell_machen.ipynb) |
| 19 | [Das Feedforward-Netz](19_das_feedforward_netz.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/19_das_feedforward_netz.ipynb) |
| 20 | [Residualverbindungen](20_residualverbindungen.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/20_residualverbindungen.ipynb) |
| 21 | [Layer Normalization](21_layer_normalization.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/21_layer_normalization.ipynb) |
| 22 | [Dropout und die anderen Regularisierer](22_dropout_und_die_anderen_regularisierer.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/22_dropout_und_die_anderen_regularisierer.ipynb) |
| 23 | [Einen Transformer-Block zusammenbauen](23_einen_transformer_block_zusammenbauen.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/23_einen_transformer_block_zusammenbauen.ipynb) |
| 24 | [Die ursprüngliche Encoder-Decoder-Architektur](24_die_urspruengliche_encoder_decoder_architektur.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/24_die_urspruengliche_encoder_decoder_architektur.ipynb) |
| 25 | [Reine Encoder-Modelle](25_reine_encoder_modelle.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/25_reine_encoder_modelle.ipynb) |
| 26 | [Reine Decoder-Modelle](26_reine_decoder_modelle.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/26_reine_decoder_modelle.ipynb) |
| 27 | [Von Logits zu Text](27_von_logits_zu_text.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/27_von_logits_zu_text.ipynb) |
| 28 | [Das Ganze in NumPy bauen](28_das_ganze_in_numpy_bauen.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/28_das_ganze_in_numpy_bauen.ipynb) |
| 29 | [Dasselbe Modell in PyTorch](29_dasselbe_modell_in_pytorch.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/29_dasselbe_modell_in_pytorch.ipynb) |
| 30 | [Einen kleinen Transformer von Anfang bis Ende trainieren](30_einen_kleinen_transformer_von_anfang_bis_ende_trainieren.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/30_einen_kleinen_transformer_von_anfang_bis_ende_trainieren.ipynb) |
| 31 | [Parameter, Rechenaufwand und Maßstab](31_parameter_rechenaufwand_und_massstab.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/31_parameter_rechenaufwand_und_massstab.ipynb) |
| 32 | [Fine-Tuning und Anpassung](32_fine_tuning_und_anpassung.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/32_fine_tuning_und_anpassung.ipynb) |
| 33 | [Transformer jenseits von Text](33_transformer_jenseits_von_text.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/33_transformer_jenseits_von_text.ipynb) |
| 34 | [Paper lesen ohne Angst](34_paper_lesen_ohne_angst.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/de/34_paper_lesen_ohne_angst.ipynb) |

## Lizenz

MIT. Für alles nutzbar, auch kommerziell.
