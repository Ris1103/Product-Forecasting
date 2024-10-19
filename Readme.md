# Store Item Demand Forecasting

## Project Overview

This project focuses on building a predictive model for store item demand forecasting. The goal is to predict future sales for a store's inventory using historical sales data. The notebook takes the user through a comprehensive data science workflow, including Exploratory Data Analysis (EDA), Feature Engineering, Model Building, and Evaluation.

## Dataset Overview

The dataset contains daily sales data of 50 items across 10 stores for a period of 5 years. The data used in this project is sourced from Kaggle's "Store Item Demand Forecasting Challenge."

- **Source URL**: [Kaggle Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview)

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

The EDA section explores the dataset to uncover patterns, insights, and potential issues. It includes:

- **Data Inspection**: Loaded and reviewed the structure of the dataset, including columns, data types, and missing values.
- **Visualization**: Plotted key metrics using various visualizations to understand seasonality, trends, and anomalies.
- **Insights**:
  - Sales data exhibited seasonal trends and patterns.
  - Store-level and item-level analysis highlighted differences in demand.
  - Time-series decomposition was performed to observe trend, seasonality, and residuals.

### 2. Feature Engineering

Feature engineering was a crucial step in building an effective model. In this section:

- **Date Features**: Extracted relevant features from the date, including year, month, week, day of the week.
- **Rolling Features**: Added rolling window statistics to capture the moving average of sales, which helps capture trends effectively.

### 3. Data Preprocessing

- **Scaling and Transformation**: Applied scaling where necessary, especially to the target and numerical features to enhance model performance.
- **Train-Test Split**: Data was split into training and validation sets in a time-series-aware manner to avoid data leakage and ensure robust evaluation.

### 4. Model Building

In the model-building section, various machine learning models were tested to determine which provided the most accurate forecasts.

- **Baseline Model**: Implemented a naive forecasting model as a benchmark.
- **Advanced Models**:
  - **Linear Regression**: To capture linear relationships between features and the target variable.
  - **Random Forest**: A tree-based model used to capture non-linear dependencies in the data.
  - **XGBoost**: Gradient boosting model used to achieve a more accurate forecast by capturing complex interactions between features.
- **Hyperparameter Tuning**: Performed hyperparameter tuning using grid search and cross-validation to improve model performance.

### 5. Model Evaluation

- **Evaluation Metrics**: Used RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) to evaluate model performance.
- **Results**:
  - Compared model results to the baseline to understand performance improvements.
  - XGBoost performed the best among the tested models in terms of RMSE.
- **Validation Plots**: Created actual vs. predicted sales plots to visually assess model accuracy.

### 6. Conclusion and Future Work

- **Conclusion**: The XGBoost model proved to be the most effective in forecasting sales demand, significantly outperforming the baseline and other models.
- **Future Work**: Potential improvements include:
  - Incorporating additional exogenous features, such as economic indicators or promotional campaigns.
  - Trying out deep learning models like LSTMs for capturing sequential dependencies more effectively.

## Files in Repository

- **Product_forcasting.ipynb**: The Jupyter notebook containing the full analysis, from EDA to model evaluation.
- **submission.csv**: Final submission file containing the predicted values.

## Dependencies

To run the notebook, you need the following libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

## Acknowledgements

The dataset is provided by Kaggle for the "Store Item Demand Forecasting Challenge".
