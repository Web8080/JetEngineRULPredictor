# JetEngineRULPredictor
This repository contains the complete code and documentation for predicting the Remaining Useful Life (RUL) of engines using the NASA CMAPSS Turbofan Jet Engine dataset. The project leverages advanced machine learning techniques, including XGBoost and LSTM (Long Short-Term Memory), to model engine degradation based on time-series sensor data.

## Overview

This project predicts the Remaining Useful Life (RUL) of engines from the NASA Turbofan Jet Engine dataset. It uses advanced machine learning models such as **XGBoost** and **LSTM**  to enhance adaptability to new engines.

The project also leverages feature engineering techniques like **PCA** for dimensionality reduction and **time-series feature extraction** for building robust models. The model is explainable using **SHAP** for understanding the contribution of each feature to the RUL prediction.


Requirements
* Python 3.8+
* TensorFlow
* XGBoost
* Scikit-learn
* SHAP

Dataset
The dataset used in this project is from the NASA CMAPSS dataset. It contains multivariate time-series sensor data, operational settings, and truncated cycles representing the engine's operational status.


Features
* Operational Settings: 3 continuous variables representing different operational settings of the engine.
* Sensor Measurements: 21 continuous variables representing different sensor readings for the engine.
* Remaining Useful Life (RUL): The target variable, calculated as the number of cycles remaining before engine failure.


Key models and Techniques 
- XGBoost: Non-linear regression model for predicting RUL.
- LSTM: Recurrent neural network to model time-series dependencies.
- Feature Engineering: Rolling statistics, lag features, and dimensionality reduction (PCA).
- Model Explainability: SHAP values for feature contribution interpretation.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/nasa-turbofan-rul.git
cd nasa-turbofan-rul
pip install -r requirements.txt
```

## How to Use

### Data Preprocessing
Ensure that the input data is scaled and transformed the same way as during training:
```python
# Load and preprocess data
from sklearn.preprocessing import MinMaxScaler
scaler = joblib.load('scaler.pkl')
new_data = pd.read_csv('new_engine_data.csv')
scaled_data = scaler.transform(new_data)
```

### Load the Pre-Trained Model
```python
import joblib
xgb_model = joblib.load('xgboost_rul_model.pkl')
```

### Make Predictions
```python
predicted_rul = xgb_model.predict(scaled_data)
print(f"Predicted RUL: {predicted_rul}")
```

### Model Explainability
Use SHAP to interpret the model's predictions:
```python
import shap
explainer = shap.Explainer(xgb_model)
shap_values = explainer.shap_values(scaled_data)
shap.summary_plot(shap_values, scaled_data)
```

## Project Structure

```
├── data
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
├── models
│   ├── xgboost_model.pkl
    ├── xgboost_model.pickle
│   ├── scaler.pkl
    ├── pca.pkl
    ├── lstm_model.h5
    ├── lstm_model.pickle.pkl
│   ├── preprocessing_pipeline.pkl
├── notebooks
│   ├── code.ipynb
├── README.md
└── requirements.txt
```

## Results

- Best LSTM RMSE: 48.60
- Best LSTM MAE**: 39.04

-XGBoost Tuned RMSE: 42.25
-XGBoost Tuned MAE: 32.50

## Future Work

- Implement **meta-learning** (MAML) to enhance model adaptability to new engines.
- Deploy the model on a cloud platform with real-time RUL prediction.
- Integrate with dashboard visualization for tracking engine health.


