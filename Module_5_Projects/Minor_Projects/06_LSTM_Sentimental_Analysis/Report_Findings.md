# LSTM Sentiment Analysis: Report

## Setup

IMDB movie reviews, binary sentiment (positive/negative). Full custom pipeline: lowercase, remove HTML/URLs/punctuation, remove stopwords, stem, build a 5000 word vocabulary from training data only, encode and pad to a fixed length of 250 tokens. Train/validation/test split applied properly: validation carved from training data for checkpointing, test set touched exactly once, at final evaluation.

Architecture: `nn.Embedding(vocab_size, 100) -> nn.LSTM(100, 128, num_layers=2, batch_first=True) -> Dropout(0.3) -> Linear(128, 1)`. Trained for 40 epochs, `BCEWithLogitsLoss`, Adam.

## Results

Best checkpoint: epoch 29 of 40.

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Training | 93.34% | 92.43% | 94.46% | 93.43% |
| **Testing** | **83.12%** | **82.46%** | **84.31%** | **83.37%** |

## Findings

**1. The LSTM genuinely learned the task, and the RNN comparison shows exactly why that mattered.** A plain RNN trained on these same 250 token sequences never moved off a loss value near what random guessing produces, the textbook vanishing gradient signature at that sequence length (see `deep-learning/rnn-sentiment` for the full failing run). Swapping the recurrent layer from `nn.RNN` to `nn.LSTM`, with no other change, took the model from unable to learn at all to a genuinely working classifier. This is the clearest, most concrete demonstration in this whole project of why LSTM exists as an architecture, not just a theoretical claim from the notes, but a direct, reproducible before/after on the same data.

**2. There is a real, worth-naming gap between training and test performance** (93.3% vs 83.1% accuracy, roughly 10 points). Some of this is expected and healthy, the model is meant to fit training data closely, some of it reflects genuine overfitting, visible in the loss curve as validation loss climbing past roughly epoch 29 while training loss kept falling, which is exactly why checkpointing at the best validation epoch (29, not 40) matters here rather than just using the final epoch's weights.

**3. 83% test accuracy on IMDB sentiment is a solid, credible result for a from-scratch pipeline.** This is not using pretrained embeddings (GloVe, Word2Vec) or a pretrained language model, every part of the vocabulary and every word vector was learned from this dataset alone, starting from random initialization. Results in this range are consistent with what a properly built, non-pretrained LSTM sentiment classifier is expected to achieve on this dataset.

## Next steps

- Try `pos_weight` or class balancing if precision/recall show meaningful asymmetry on closer inspection.
- Try pretrained embeddings (GloVe) as a follow-up comparison: same architecture, embeddings frozen or fine-tuned, to see how much of the current gap to state-of-the-art is attributable to learning embeddings from scratch on a relatively small dataset.
- Try a lower dropout or `num_layers=1` to see whether the overfitting gap narrows without sacrificing test accuracy, given `num_layers=2` here is a deliberately heavier configuration than the `num_layers=1` used in earlier smaller builds.
