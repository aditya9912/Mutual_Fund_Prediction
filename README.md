# Mutual Fund Performance Prediction Using Machine Learning

## Overview
This project focuses on predicting mutual fund returns using machine learning techniques (Random Forest and XGBoost), along with SHAP-based interpretability and clustering to segment funds by risk. The goal is to assist investors in selecting high-performing mutual funds based on data-driven insights.

## Author
Aditya Bindenganavale Dwajan**  
MSc Data Science, University of Hertfordshire  
Student ID: 23005333

## GitHub Repository
[GitHub Repository](https://github.com/aditya9912/Mutual_Fund_Prediction)

---

## Table of Contents
- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## Project Objectives
- Predict mutual fund returns using machine learning.
- Evaluate performance using RMSE and R².
- Analyse feature importance using SHAP values.
- Segment funds into risk profiles using K-Means clustering.

---

## Technologies Used
- Python
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `optuna`

---

## Dataset
- Source:Morningstar via Kaggle  
- Link:[Morningstar European Mutual Funds](https://www.kaggle.com/datasets/stefanoleone992/european-funds-dataset-from-morningstar)  
- Over 57,000 mutual funds from 2007–2017, with features like quarterly returns, ratings, expense ratios, volatility, and more.

---

## Installation
```bash
# Clone the repository
git clone https://github.com/aditya9912/Mutual_Fund_Prediction.git
cd Mutual_Fund_Prediction

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

---

## Usage
1. Download the dataset from Kaggle and place it in the project root directory.
2. Open the Jupyter Notebook directly in your browser:  
   [aditya_final_project.ipynb](https://github.com/aditya9912/Mutual_Fund_Prediction/blob/main/aditya_final_project.ipynb)
3. Run all cells step-by-step to:
   - Preprocess the data
   - Train models (Random Forest and XGBoost)
   - Evaluate model performance
   - Interpret predictions using SHAP
   - Cluster funds into Aggressive, Balanced, and Conservative profiles

---

## Results
| Model               | RMSE     | R² Score |
|--------------------|----------|----------|
| Random Forest       | 0.3167   | 0.9875   |
| XGBoost (Tuned)     | 0.2084| 0.9946|

- Best Model:Tuned XGBoost
- Key Features:fund_return_2016, return-to-volatility ratio
- Clustering:3 risk categories identified using K-Means

---

## Future Work
- Integrate macroeconomic indicators (e.g., inflation, interest rates).
- Deploy the model for real-time predictions.
- Explore time-series models (e.g., LSTM) for dynamic forecasting.

---

## License
This project is for academic purposes only. For other uses, please contact the author.
