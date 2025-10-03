# Introvert vs Extrovert Predictor

Predict whether a person is an **Introvert** or an **Extrovert** based on social behavior and personality traits using multiple machine learning models.

## Dataset

This dataset is part of the [Kaggle Playground Series 2025 competition](https://www.kaggle.com/competitions/playground-series-s5e7/data).

**Try the Streamlit App here:** [Introvert vs Extrovert Predictor](https://personality-test-extrovert-vs-introvert.streamlit.app/)


Features:

- **Time_spent_Alone** → Hours spent alone.
- **Stage_fear** → Stage fear (Yes/No).
- **Social_event_attendance** → How often they attend social events.
- **Going_outside** → Frequency of going outside.
- **Drained_after_socializing** → Feeling drained after socializing (Yes/No).
- **Friends_circle_size** → Number of close friends.
- **Post_frequency** → How often they post online.
- **Personality** → Target column (Introvert/Extrovert).

## Preprocessing

- Label encoding for categorical features.
- Iterative Imputer for missing values.
- Scaling: StandardScaler, RobustScaler, PowerTransformer (Yeo-Johnson).
- Outliers handled in `Time_spent_Alone`.

## Models Trained

- Logistic Regression  
- K-Nearest Neighbors  
- Decision Tree  
- Random Forest  
- Extra Trees  
- Gaussian Naive Bayes  
- Bagging (Decision Tree)  
- AdaBoost  
- XGBoost  
- LightGBM  
- CatBoost  

Evaluation metrics: Accuracy, Precision, Recall, F1-score.  
Confusion matrices plotted for all models.

## Best Model
The best performing model is selected automatically based on F1-score ('AdaBoost').  

## Usage

1. Clone the repository.
2. Install dependencies (`scikit-learn`, `pandas`, `numpy`, `xgboost`, `lightgbm`, `catboost`, `plotly`, `seaborn`, `matplotlib`).
3. Run the main notebook/script to preprocess data, train models, and generate predictions.

## Files

- `Files/*.pkl` → Saved models, encoders, scalers, imputers, and results.
