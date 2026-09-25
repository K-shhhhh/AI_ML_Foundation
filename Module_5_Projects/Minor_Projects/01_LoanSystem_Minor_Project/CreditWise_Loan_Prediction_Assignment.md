# Supervised ML Assignment: CreditWise Loan System

## Problem Statement
SecureTrust Bank offers personal and home loans and relies on a manual verification process to evaluate applications based on income, employment, and credit history. This manual process is time-consuming, biased, and inconsistent, leading to two major challenges:
1. Good customers are sometimes rejected (loss of business).
2. High-risk customers are sometimes approved (financial losses).

**Objective:** Design and develop an intelligent Machine Learning system that learns hidden patterns from historical records to automatically predict whether a loan should be Approved or Rejected, ensuring accurate, fast, and unbiased decisions prior to final human verification.

## Dataset Overview
The dataset contains multiple attributes describing the personal, financial, and credit information of loan applicants.

| Column | Description |
| :--- | :--- |
| Applicant_ID | Unique applicant ID |
| Applicant_Income | Monthly income of applicant |
| Coapplicant_Income | Monthly income of co-applicant |
| Employment_Status | Salaried / Self-Employed / Business |
| Age | Applicant age |
| Marital_Status | Married / Single |
| Dependents | Number of dependents |
| Credit_Score | Credit bureau score |
| Existing_Loans | Number of already running loans |
| DTI_Ratio | Debt-to-Income ratio |
| Savings | Savings balance |
| Collateral_Value | Value of collateral provided |
| Loan_Amount | Loan amount requested[cite: 11] |
| Loan_Term | Loan duration (months)[cite: 11] |
| Loan_Purpose | Home / Education / Personal / Business |
| Property_Area | Urban / Semi-Urban / Rural |
| Education_Level | Graduate / Postgraduate / Undergraduate |
| Gender | Male / Female[cite: 11] |
| Employer_Category | Govt/Private / Self |
| **Loan_Approved** | **Target (1 = Approved, 0 = Rejected)** |