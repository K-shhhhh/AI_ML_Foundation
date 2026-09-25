# Deep Learning: Sentiment Analysis using LSTM

## Project Dataset: IMDB Movie Reviews Dataset

## Problem Statement

Sentiment analysis is a core natural language processing problem: given a piece of free-form text, determine whether the writer's opinion is positive or negative. Unlike tabular or image data, text has no fixed numeric structure to begin with, it has to be cleaned, tokenized, and converted into a sequence of numbers before any model can process it, and the order those words appear in genuinely carries meaning.

The objective of this project is to build a full, from-scratch text classification pipeline: cleaning raw reviews, building a vocabulary from the training data, encoding and padding reviews into fixed-length sequences, and training a **Long Short-Term Memory network (LSTM)** to classify each review as positive or negative using the **IMDB Movie Reviews dataset**.

An LSTM was chosen over a plain Recurrent Neural Network (RNN) after directly testing both: a plain RNN trained on these same sequences failed to learn at all, its loss stayed near the value expected from random guessing across the full training run, the textbook signature of the vanishing gradient problem across a 250 word sequence. That failing RNN run is kept as its own standalone project (`deep-learning/rnn-sentiment`), specifically to demonstrate the problem LSTM exists to solve, rather than discarded once the fix was found.

The project includes text cleaning (lowercasing, HTML/URL/punctuation removal, stopword removal, stemming), custom vocabulary construction with leakage-safe train/validation/test splitting, sequence encoding and padding, LSTM model design, training with checkpointing on a genuine validation set, and evaluation using accuracy, precision, recall, and F1 score.

## Expected Outcome

A trained LSTM model capable of classifying IMDB movie review sentiment with meaningfully better than baseline accuracy, alongside a clear, working demonstration of why a plain RNN is unsuited to this task at this sequence length.
