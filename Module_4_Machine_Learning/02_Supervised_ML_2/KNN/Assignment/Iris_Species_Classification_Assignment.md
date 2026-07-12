# Supervised ML Assignment: Iris Species Classification

## Problem Statement
A botanical research center needs to automate the identification of Iris flower species. Manual measurement is time-consuming, error-prone, and not scalable. The goal is to build a machine learning classification system that predicts the species based on sepal and petal measurements.

## Objectives
*   Train three different classification models:
    *   **K-Nearest Neighbours (KNN)**
    *   **Logistic Regression**
    *   **Naive Bayes**
*   Compare the performance of these models.
*   Identify and recommend the best algorithm for the task.

## Dataset Overview
The Iris dataset is a well-balanced, clean dataset consisting of 150 samples split equally across 3 species (50 samples each).

| Feature | Description |
| :--- | :--- |
| SepalLengthCm | Length of sepal (cm) |
| SepalWidthCm | Width of sepal (cm) |
| PetalLengthCm | Length of petal (cm) |
| PetalWidthCm | Width of petal (cm) |
| **Species (Target)** | **Iris-setosa, Iris-versicolor, Iris-virginica** |

## Experimental Constraints
*   **Training Split:** To simulate a more challenging environment, train the models using only 50% of the dataset and test on the remaining 50%.