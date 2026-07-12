# Supervised ML Assignment: E-commerce Purchase Prediction

## Problem Statement
An E-commerce company wants to predict whether a visitor is likely to complete a purchase based on their session behavior. Currently, the company struggles with inefficient marketing strategies due to an inability to forecast conversion.

## Objectives
*   **Exploratory Data Analysis (EDA):** Understand user browsing patterns and session interactions.
*   **Preprocessing:** Apply necessary feature transformations for numerical and categorical data.
*   **Modeling:** Develop a **Decision Tree-based classification model**.
*   **Optimization:** Apply **pruning** to improve the model's performance and prevent overfitting.
*   **Evaluation:** Since the dataset is imbalanced, use the **F1 score** as the primary metric. The benchmark for effectiveness is an **F1 score of 0.55**.

## Dataset Overview
The dataset includes 12,330 individual user sessions collected over one year

| Feature Name | Description |
| :--- | :--- |
| Administrative / Duration | Pages/Time spent on account management |
| Informational / Duration | Pages/Time spent on FAQ, policies, etc. |
| ProductRelated / Duration | Pages/Time spent on products |
| Bounce Rates | Percentage of visitors leaving after one page |
| ExitRates | Percentage of exits from a page |
| PageValues | Average value of pages visited before purchase |
| SpecialDay | Closeness to special days (e.g., Valentine's) |
| Month | Month of the visit |
| Operating Systems | Encoded OS |
| Browser | Encoded Browser |
| Region | Geographic region |
| TrafficType | Source of traffic |
| VisitorType | Returning/New/Other |
| Weekend | Whether visit occurred on a weekend |
| **Revenue (Target)** | **Purchase made (1 = Yes, 0 = No)** |