# football-goal-prediction

This repository contains the R code and dataset used for a report on predicting football goal scoring using resampling based methods.

## Files

- football_goal_prediction_analysis.R  is the main script used for data preparation, modelling, and evaluation
- 2021-2022 Football Player Stats.csv – dataset used in the analysis

## Methods

The models implemented include:
- Linear regression
- Poisson regression
- Bagged Poisson regression
- Gradient boosting (GBM)

Model performance is evaluated using 10-fold cross-validation and RMSE.

## How to run

1. Place the CSV file in the same folder as the R script  
2. Open the script in RStudio  
3. Run the code from top to bottom  
