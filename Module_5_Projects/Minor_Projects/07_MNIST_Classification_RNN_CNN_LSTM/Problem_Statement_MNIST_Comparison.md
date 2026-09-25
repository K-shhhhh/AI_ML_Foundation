# Deep Learning (Assignment Project)

## Handwritten Digit Classification using CNN & RNN

## Project Dataset: MNIST Digit Classification

## Problem Statement

Handwritten digit recognition is a fundamental problem in computer vision with applications in postal automation, bank cheque processing, and digitising handwritten documents. Traditional machine learning methods require manual feature extraction, which can limit performance and scalability.

The objective of this project is to develop a **Convolutional Neural Network (CNN)** and a **Recurrent Neural Network (RNN)** capable of automatically learning spatial features from handwritten images and accurately classifying digits from **0 to 9** using the **MNIST dataset**. We'll then compare the performance of both these neural networks.

The dataset contains grayscale images of handwritten digits, each of size **28 x 28 pixels**. The goal is to train a CNN model and an RNN model that can correctly classify each image into one of the ten digit classes.

The project involves designing an appropriate CNN and RNN architecture, training the model, evaluating its performance using metrics such as **accuracy and confusion matrix**, and analysing how different architectural choices affect classification performance.

**Note on scope:** an LSTM model was added as a third point of comparison alongside CNN and RNN, following directly from the RNN vs LSTM comparison established in the sentiment analysis project. Treating an image as a sequence of rows is a genuinely different regime from a 250 word review, at only 28 timesteps, so this project also tests whether the RNN failure seen on long text sequences reproduces at a much shorter sequence length, or whether a plain RNN remains viable when the vanishing gradient problem has far less distance to compound over.

## Expected Outcome

Trained CNN, RNN, and LSTM models capable of recognising handwritten digits with high accuracy (typically greater than 95%), with a direct, evidence-based comparison of how a spatial architecture (CNN) and two sequence architectures (RNN, LSTM) perform on the same task.
