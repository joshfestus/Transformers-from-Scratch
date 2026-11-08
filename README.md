# Transformers from Scratch

The code from *Transformers from Scratch: A Beginner's Guide to the Neural
Network Behind Large Language Models*, by Josh Festus.

Every listing in the book comes from these files. The book executes them when
it is typeset, so what is printed on the page is what these files actually
print. The notebooks are generated from the same files.

German readers: the notebooks for the German edition, *Transformer von
Grund auf*, are in [`notebooks/de`](notebooks/de), with the German code
they use. Deutsche Leser finden die Notebooks zur deutschen Ausgabe im
Ordner [`notebooks/de`](notebooks/de).

## Running it

Each chapter has a notebook that opens in Colab and needs nothing installed.
Click a badge below. The first cell fetches the two or three files that
chapter uses and the rest run in order.

To run it locally instead:

```
git clone https://github.com/joshfestus/Transformers-from-Scratch
cd Transformers-from-Scratch
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python code/ch14_attention.py
```

Nothing here needs a graphics card. The longest thing in the repository is the
training run in chapter 30, which takes about ninety seconds on a laptop.

## The chapters

| Chapter | Notebook | |
|---|---|---|
| 1 | [What a Neural Network Actually Is](notebooks/01_what_a_neural_network_actually_is.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/01_what_a_neural_network_actually_is.ipynb) |
| 2 | [Vectors, Matrices, and the Language of Shapes](notebooks/02_vectors_matrices_and_the_language_of_shapes.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/02_vectors_matrices_and_the_language_of_shapes.ipynb) |
| 3 | [Matrix Multiplication as Information Flow](notebooks/03_matrix_multiplication_as_information_flow.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/03_matrix_multiplication_as_information_flow.ipynb) |
| 4 | [Why Stacking Linear Layers Is Pointless](notebooks/04_why_stacking_linear_layers_is_pointless.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/04_why_stacking_linear_layers_is_pointless.ipynb) |
| 5 | [Softmax, and How Scores Become Probabilities](notebooks/05_softmax_and_how_scores_become_probabilities.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/05_softmax_and_how_scores_become_probabilities.ipynb) |
| 6 | [How a Network Learns](notebooks/06_how_a_network_learns.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/06_how_a_network_learns.ipynb) |
| 7 | [Gradient Descent, Learning Rates and Optimizers](notebooks/07_gradient_descent_learning_rates_and_optimizers.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/07_gradient_descent_learning_rates_and_optimizers.ipynb) |
| 8 | [Tokenization](notebooks/08_tokenization.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/08_tokenization.ipynb) |
| 9 | [Embeddings](notebooks/09_embeddings.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/09_embeddings.ipynb) |
| 10 | [Position, the Information Attention Throws Away](notebooks/10_position_the_information_attention_throws_away.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/10_position_the_information_attention_throws_away.ipynb) |
| 11 | [The Problem Attention Was Invented to Solve](notebooks/11_the_problem_attention_was_invented_to_solve.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/11_the_problem_attention_was_invented_to_solve.ipynb) |
| 12 | [Query, Key and Value](notebooks/12_query_key_and_value.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/12_query_key_and_value.ipynb) |
| 13 | [Scaled Dot-Product Attention, Symbol by Symbol](notebooks/13_scaled_dot_product_attention_symbol_by_symbol.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/13_scaled_dot_product_attention_symbol_by_symbol.ipynb) |
| 14 | [A Complete Numerical Walkthrough](notebooks/14_a_complete_numerical_walkthrough.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/14_a_complete_numerical_walkthrough.ipynb) |
| 15 | [Multi-Head Attention](notebooks/15_multi_head_attention.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/15_multi_head_attention.ipynb) |
| 16 | [Masking](notebooks/16_masking.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/16_masking.ipynb) |
| 17 | [Cross-Attention](notebooks/17_cross_attention.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/17_cross_attention.ipynb) |
| 18 | [Making Attention Fast](notebooks/18_making_attention_fast.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/18_making_attention_fast.ipynb) |
| 19 | [The Feedforward Network](notebooks/19_the_feedforward_network.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/19_the_feedforward_network.ipynb) |
| 20 | [Residual Connections](notebooks/20_residual_connections.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/20_residual_connections.ipynb) |
| 21 | [Layer Normalization](notebooks/21_layer_normalization.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/21_layer_normalization.ipynb) |
| 22 | [Dropout and the Other Regularizers](notebooks/22_dropout_and_the_other_regularizers.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/22_dropout_and_the_other_regularizers.ipynb) |
| 23 | [Assembling One Transformer Block](notebooks/23_assembling_one_transformer_block.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/23_assembling_one_transformer_block.ipynb) |
| 24 | [The Original Encoder-Decoder Architecture](notebooks/24_the_original_encoder_decoder_architecture.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/24_the_original_encoder_decoder_architecture.ipynb) |
| 25 | [Encoder-Only Models](notebooks/25_encoder_only_models.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/25_encoder_only_models.ipynb) |
| 26 | [Decoder-Only Models](notebooks/26_decoder_only_models.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/26_decoder_only_models.ipynb) |
| 27 | [From Logits to Text](notebooks/27_from_logits_to_text.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/27_from_logits_to_text.ipynb) |
| 28 | [Building the Whole Thing in NumPy](notebooks/28_building_the_whole_thing_in_numpy.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/28_building_the_whole_thing_in_numpy.ipynb) |
| 29 | [The Same Model in PyTorch](notebooks/29_the_same_model_in_pytorch.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/29_the_same_model_in_pytorch.ipynb) |
| 30 | [Training a Small Transformer End to End](notebooks/30_training_a_small_transformer_end_to_end.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/30_training_a_small_transformer_end_to_end.ipynb) |
| 31 | [Parameters, Compute and Scale](notebooks/31_parameters_compute_and_scale.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/31_parameters_compute_and_scale.ipynb) |
| 32 | [Fine-Tuning and Adaptation](notebooks/32_fine_tuning_and_adaptation.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/32_fine_tuning_and_adaptation.ipynb) |
| 33 | [Transformers Beyond Text](notebooks/33_transformers_beyond_text.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/33_transformers_beyond_text.ipynb) |
| 34 | [Reading Papers Without Fear](notebooks/34_reading_papers_without_fear.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshfestus/Transformers-from-Scratch/blob/main/notebooks/34_reading_papers_without_fear.ipynb) |

## Two files worth starting with

`code/transformer_numpy.py` is a complete transformer in about a hundred lines
with no framework anywhere. `code/transformer_torch.py` is the same model in
PyTorch, and running it loads identical weights into both and checks that they
agree to fifteen decimal places.

## Licence

MIT. Use it for anything, including commercially.
