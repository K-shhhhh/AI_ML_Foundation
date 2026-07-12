# Data Visualization Assignment

## Q1: NumPy Random Distribution
For the following dataset generated using NumPy: `np.random.normal(loc=70, scale=10, size=200)`

*   Plot a histogram of the generated scores.
*   Draw a vertical line representing the mean and interpret the resulting distribution.

## Q2: Seaborn Penguins Dataset
Using the Seaborn penguins dataset, perform the following:

*   Plot `flipper_length_mm` vs `body_mass_g`.
*   Use color to distinguish between different species.

## Q3: Seaborn Tips Dataset (Box Plot)
Using the Seaborn tips dataset, perform the following:

*   Create a box plot of `total_bill` for each day.
*   Add `hue="sex"` to distinguish by gender.
*   Identify which day has the highest median bill.
*   Comment on any observed outliers.

## Q4: Seaborn Tips Dataset (Scatter Plot)
Using the Seaborn tips dataset, perform the following:

*   Create a scatter plot of `total_bill` vs `tip`.
*   Color points by `sex`.
*   Use different markers for `smoker` status.
*   Add a title and appropriate axis labels.
*   **Analysis:** Do smokers tend to tip differently than non-smokers?

## Q5: Seaborn Tips Dataset (Heatmap)
Using the Seaborn tips dataset, perform the following:

*   Create a pivot table of the average tip where:
    *   **Rows:** `day`
    *   **Columns:** `time`
*   Plot the pivot table as a heatmap.
*   Annotate the heatmap with the average tip values.
*   **Interpretation:** Which day-time combination results in the highest average tip?