MODEL_DESCRIPTIONS = {
    "LinearRegression": "Linear Regression: Fits a straight line to minimize squared error. Fast, interpretable, but can't model non-linearities.",
    "RidgeCV": "Ridge Regression (CV): Linear model with L2 regularization. Reduces overfitting, especially with many features.",
    "Lasso": "Lasso Regression: Linear model with L1 regularization. Can shrink some coefficients to zero (feature selection).",
    "ElasticNet": "ElasticNet: Mix of L1 and L2 regularization. Useful when there are multiple correlated features.",
    "RandomForest": "Random Forest: Ensemble of decision trees. Handles non-linearities, robust to outliers, less interpretable.",
    "HistGB": "Histogram-based Gradient Boosting: Fast, scalable boosting method. Excels on large/tabular data.",
    "GradientBoostingRegressor": "Gradient Boosting: Builds trees sequentially to correct previous errors. Powerful but can overfit if not tuned.",
    "SVR": "Support Vector Regression: Finds a margin of tolerance. Good for small/medium datasets, sensitive to scaling.",
    "KNeighborsRegressor": "K-Nearest Neighbors: Predicts by averaging closest samples. Simple, non-parametric, can be slow on large data.",
    "XGBoost": "XGBoost: Highly optimized gradient boosting. State-of-the-art for many tabular ML tasks. Handles missing data, regularization, and parallelization.",
}
