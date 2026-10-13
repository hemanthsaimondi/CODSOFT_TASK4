# Task 4: Sales Prediction using Python

Predicts product sales from advertising spend on TV, Radio and Newspaper.
Dataset: ISLR Advertising dataset (200 markets, no missing values or duplicates). Sales in thousands of units, spend in thousands of dollars.

## Approach
1. **EDA**: scatter plots with trend lines and a correlation matrix (`scatter_vs_sales.png`, `correlation.png`)
2. **Statistical check** (statsmodels OLS): TV and Radio are significant; Newspaper is not (p = 0.70)
3. **Models compared** (5-fold CV, R²): simple linear regression (TV only), multiple linear regression, polynomial regression (degree 2), Random Forest, Gradient Boosting
4. **Evaluation**: 80/20 hold-out test set (40 markets)

## Results (test set)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Simple LR (TV only) | 2.444 | 3.194 | 0.677 |
| Multiple LR (all 3) | 1.461 | 1.782 | 0.899 |
| Multiple LR (TV + Radio) | 1.444 | 1.771 | 0.901 |
| **Polynomial deg 2 (TV + Radio)** | **0.474** | **0.615** | **0.988** |
| Random Forest | 0.663 | 0.791 | 0.980 |
| Gradient Boosting | 0.649 | 0.753 | 0.982 |

## Key insights
- TV is the strongest single driver of sales, Radio is next, and Newspaper adds almost nothing.
- TV and Radio work better together than separately (interaction effect), which is why the degree-2 polynomial beats plain linear regression.

## Run
```
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
python sales_prediction.py
```
