# AI & Machine Learning Foundations

A structured record of my AI and machine learning curriculum, from data analysis and the maths behind ML to deep learning and reinforcement learning. Each module holds the notebooks written while learning a topic, and the projects folder holds the end to end builds that apply it.

The repository is organised in learning order, so it can be read from Module 1 to Module 7 as one path.

---

## Highlights

| Project | What it shows | Result |
|---|---|---|
| [Mario with Double DQN](Module_5_Projects/Minor_Projects/10_Mario_Double_DQN) | An agent that learns Super Mario Bros World 1-1 from raw pixels | Clears the level every greedy run after 7.56M training steps |
| [NovaGen Health Risk Classification](Module_5_Projects/Minor_Projects/02_Novagen_Research_Minor_Project) | Pipelines, GridSearchCV and model comparison on 9,800 health records | 95.4% test accuracy with CatBoost |
| [CreditWise Loan Prediction](Module_5_Projects/Minor_Projects/01_LoanSystem_Minor_Project) | Feature engineering and regularisation for loan approval | 87.5% accuracy, F1 0.80 |
| [MNIST: CNN vs RNN vs LSTM](Module_5_Projects/Minor_Projects/07_MNIST_Classification_RNN_CNN_LSTM) | The same task solved by three architectures | Side by side report of all three |

---

## Repository Structure & Curriculum

### 🔹 Module 1: Data Analysis & Exploratory Data Analysis
Data ingestion, manipulation, web scraping and visualisation.
*   **Environment Foundations:** Jupyter setup and workflow.
*   **Numerical Computing:** Array operations and vectorisation with NumPy.
*   **Structured Data:** Cleaning, alignment and transformation with Pandas.
*   **Data Collection:** Web scraping and REST APIs, with the scraped and cleaned datasets.
*   **Visualisation:** Distributions, relationships and categorical analysis with Matplotlib and Seaborn.

### 🔹 Module 2: Relational Databases & SQL
*   **Database Management:** Schema design, transactions, ACID properties and indexing.
*   **Analytical Querying:** Joins, subqueries, aggregations and window functions.

### 🔹 Module 3: Mathematical Foundations for ML
*   **Probability:** Distributions, conditional probability, expectation, variance and inference.
*   **Calculus:** Partial derivatives, the chain rule and the gradients behind optimisation.

### 🔹 Module 4: Machine Learning
**Supervised learning**
*   **From scratch with NumPy:** Linear Regression, Logistic Regression, Naive Bayes and KNN.
*   **With Scikit-Learn:** Linear and Lasso Regression, classification, KNN, Naive Bayes, Support Vector Machines, Decision Trees with pruning, and Ensemble Learning.

**Unsupervised learning**
*   **Clustering:** K-Means, Hierarchical Clustering and DBSCAN.
*   **Anomaly Detection:** Techniques for finding unusual points in data.

### 🔹 Module 5: Projects
End to end builds that apply the modules, from classical ML to reinforcement learning.

| # | Project | Area | Description |
|---|---|---|---|
| 01 | [CreditWise Loan Prediction](Module_5_Projects/Minor_Projects/01_LoanSystem_Minor_Project) | Supervised ML | Predicts loan approval from applicant income, credit history and collateral, comparing Logistic Regression variants, Naive Bayes and KNN |
| 02 | [NovaGen Health Risk Classification](Module_5_Projects/Minor_Projects/02_Novagen_Research_Minor_Project) | Supervised ML | Classifies participants as healthy or at higher risk, with pipelines, cross validated tuning and a comparison of tree based models |
| 03 | [SmartCart Customer Segmentation](Module_5_Projects/Minor_Projects/03_SmartCart_Mini_Project) | Unsupervised ML | Clusters 2,240 e-commerce customers by demographics and purchase behaviour for targeted marketing |
| 04 | [ANN vs Classical ML](Module_5_Projects/Minor_Projects/04_ANN_vs_ML_Comparision) | Deep Learning | A controlled comparison of an ANN against Linear or Logistic Regression, Random Forest and Gradient Boosting on synthetic regression, binary and multiclass data |
| 05 | [Cats vs Dogs CNN](Module_5_Projects/Minor_Projects/05_CNN_Cat_Dog_Classification) | Deep Learning | Binary image classification with a Convolutional Neural Network in PyTorch |
| 06 | [LSTM Sentiment Analysis](Module_5_Projects/Minor_Projects/06_LSTM_Sentimental_Analysis) | Deep Learning | Classifies IMDB reviews as positive or negative, with a function to predict on new sentences |
| 07 | [MNIST: CNN vs RNN vs LSTM](Module_5_Projects/Minor_Projects/07_MNIST_Classification_RNN_CNN_LSTM) | Deep Learning | Handwritten digit classification with three architectures under the same train, validation and test split |
| 08 | [Taxi: SARSA vs Q Learning](Module_5_Projects/Minor_Projects/08_Taxi_v4_SARSA_vs_Q_Learning) | Reinforcement Learning | Compares on policy SARSA with off policy Q learning on Gymnasium's Taxi environment |
| 09 | [Blackjack with Dynamic Programming](Module_5_Projects/Minor_Projects/09_Blackjack_Dynamic_Programming) | Reinforcement Learning | Derives the transition probabilities of Blackjack and solves for the best policy with value iteration |
| 10 | [Mario with Double DQN](Module_5_Projects/Minor_Projects/10_Mario_Double_DQN) | Reinforcement Learning | Learns World 1-1 from raw pixels, with resumable training, evaluation, plots and a script to watch the trained agent |

### 🔹 Module 6: Deep Learning
Neural networks built in PyTorch.
*   **ANN:** Regression on power plant data and classification on date fruit data, each with a saved best model.
*   **CNN:** Convolution, pooling and image classification.
*   **RNN:** A plain RNN for sentiment analysis, kept to show the vanishing gradient problem that LSTMs solve.

### 🔹 Module 7: Reinforcement Learning
One short walkthrough per method. Each one leads into a bigger project in Module 5.

| Method | Walkthrough | Project built on it |
|---|---|---|
| Temporal Difference | [Cliff Walking](Module_7_Reinforcement_Learning/01_Temporal_Difference/Cliff_Walking) (SARSA and Q learning) | 08 Taxi |
| Dynamic Programming | [Frozen Lake](Module_7_Reinforcement_Learning/02_Dynamic_Programming/Frozen_Lake) (value iteration) | 09 Blackjack |
| Monte Carlo | [Blackjack](Module_7_Reinforcement_Learning/03_Monte_Carlo/Blackjack_Monte_Carlo) (learning from complete episodes) | Bootstrap simulation on a trading backtest (upcoming) |
| Deep Q Networks | [Flappy Bird DQN](Module_7_Reinforcement_Learning/04_Deep_Q_Networks/Flappy_Bird_DQN) | 10 Mario Double DQN |

---

## Upcoming

*   **Monte Carlo Bootstrap Simulation:** Resampling the trades of a backtested trading strategy to measure the spread of possible outcomes and drawdowns.
*   **Major Projects:** Larger end to end systems in the `Major_Projects` folder.

---

## Tech Stack & Tooling
*   **Languages:** Python (Jupyter Notebooks and scripts), SQL (MySQL)
*   **Data Science:** NumPy, Pandas, Scikit-Learn, CatBoost
*   **Deep Learning:** PyTorch
*   **Reinforcement Learning:** Gymnasium, Gym, gym-super-mario-bros
*   **Visualisation:** Matplotlib, Seaborn
*   **Data Collection:** Beautiful Soup, Requests (REST APIs)

---

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/K-shhhhh/AI_ML_Foundation.git
   cd AI_ML_Foundation
   ```
2. Open any module notebook in Jupyter:
   ```bash
   jupyter notebook
   ```
3. Projects with their own setup (such as Mario) list their exact requirements and commands in their own README.

Datasets are not included in the repository. Each project's README or notebook says which dataset it uses.
