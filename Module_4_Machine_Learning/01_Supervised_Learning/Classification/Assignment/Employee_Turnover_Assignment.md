# Supervised ML Assignment: Employee Turnover Prediction

## Problem Statement
A Private Company is experiencing a rising number of employee resignations. The goal is to build an intelligent ML system to predict whether an employee is likely to leave the company based on factors like job satisfaction, salary, age, and work-life balance.

## Objectives
1.  **Baseline Model:** Build a Logistic Regression model to predict `Employee_Turnover`.
2.  **Regularization:** Improve the model performance using L1 (Lasso) and L2 (Ridge) regularization.
3.  **Performance Analysis:** Compare the models and provide a recommendation for the best approach.

## Dataset Overview
The dataset contains 900 rows and 15 columns (features)

| Feature | Description |
| :--- | :--- |
| Job_Satisfaction | Level of satisfaction with the job |
| Performance_Rating | Employee performance score |
| Years_At_Company | Number of years worked in company |
| Work_Life_Balance | Balance between work and personal life |
| Distance_From_Home | Distance of home from workplace |
| Monthly_Income | Monthly salary |
| Education_Level | Education qualification level |
| Age | Age of employee |
| Num_Companies_Worked | Number of companies worked previously |
| Employee_Role | Encoded job role |
| Annual_Bonus | Bonus received annually |
| Training_Hours | Training hours attended |
| Department | Encoded department |
| Annual_Bonus_Squared | Engineered feature (bonus²) |
| Annual_Bonus_Training_Hours_Interaction | Interaction feature |
| **Employee_Turnover** | **Target (1 = Left, 0 = Stayed)** |