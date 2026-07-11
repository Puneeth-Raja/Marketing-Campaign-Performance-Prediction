# 📈 Marketing Campaign Performance Prediction

An end-to-end Machine Learning project that predicts **campaign revenue** using Regression and **campaign profitability (Profit/Loss)** using Classification. The project also includes an interactive **Streamlit web application** for real-time predictions and campaign analysis.

---

## 🚀 Project Overview

Marketing teams invest significant budgets across multiple advertising channels. Predicting campaign performance before making business decisions can help optimize spending and improve return on investment (ROI).

This project uses historical marketing campaign data to:

- Predict campaign revenue
- Classify campaigns as **Profit** or **Loss**
- Calculate campaign ROI
- Provide an easy-to-use Streamlit dashboard for predictions

---

## 🎯 Objectives

- Build a Revenue Prediction Model using Regression.
- Build a Profit/Loss Prediction Model using Classification.
- Compare multiple Machine Learning algorithms.
- Deploy the best-performing models using Streamlit.
- Help businesses make data-driven marketing decisions.

---

# 📂 Project Structure

```
Marketing Campaign Performance Prediction
│
├── app.py
│
├── Dataset
│   ├── Cleaned_data.csv
│   ├── Cleaned_data.xls
│   ├── Data_for_modelling.xls
│   ├── nykaa_campaign_data_with_nulls.csv
│   ├── purplle_campaign_data_with_nulls.csv
│   └── tira_campaign_data_with_nulls.csv
│
├── notebook
│   ├── Data Cleaning & Preprocessing.ipynb
│   ├── Exploratory Data Analysis.ipynb
│   ├── Feature Engineering.ipynb
│   └── Model.ipynb
│
├── model
│   ├── revenue_model.pkl
│   ├── profit_classifier.pkl
│   ├── Gradient Boosting.pkl
│   ├── Logistic Regression.pkl
│   └── Model Building.ipynb
│
└── README.md
```

---

# 📊 Dataset

The dataset contains marketing campaign information collected from multiple campaigns.

### Features

- Campaign Type
- Target Audience
- Duration
- Channel Used
- Conversion Rate
- Acquisition Cost
- ROI
- Impressions
- Clicks
- Conversions
- Engagement Score
- Customer Segment
- Location
- Language
- Budget
- Revenue

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Removed duplicate records
- Handled missing values
- Fixed inconsistent data
- Converted categorical features
- Feature Encoding
- Feature Engineering
- Train-Test Split

---

# 📈 Exploratory Data Analysis

Performed:

- Missing Value Analysis
- Distribution Analysis
- Correlation Analysis
- Campaign Performance Analysis
- Channel-wise Analysis
- Budget vs Revenue
- ROI Analysis
- Conversion Analysis

Libraries used:

- Pandas
- NumPy
- Matplotlib
- Plotly
- Seaborn

---

# ⚙️ Feature Engineering

Additional features were created to improve prediction performance.

Examples include:

- Total Acquisition Cost
- Net Profit
- ROI
- Profit/Loss Label

---

# 🤖 Machine Learning Models

## Regression Models

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor ✅ (Best Model)

### Regression Evaluation Metrics

- MAE
- RMSE
- R² Score

---

## Classification Models

- Logistic Regression ✅
- Random Forest Classifier
- Gradient Boosting Classifier

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score

---

# 📊 Model Performance

## Revenue Prediction

| Metric | Score |
|---------|-------|
| MAE | 14,829 |
| RMSE | 23,074.80 |
| R² Score | **0.9975** |

---

## Profit/Loss Classification

| Metric | Score |
|---------|-------|
| Accuracy | 99.67% |
| Precision | 99.70% |
| Recall | 100% |
| F1 Score | 99.64% |

---

# 🖥️ Streamlit Application

The application allows users to:

- Enter campaign details
- Predict Revenue
- Predict Profit/Loss
- Calculate ROI
- Display Prediction Results
- View Model Performance

---

# 🛠️ Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn
- Plotly

### Machine Learning

- Scikit-learn

### Deployment

- Streamlit

### Model Serialization

- Pickle

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/Puneeth-Raja/Marketing-Campaign-Performance-Prediction.git
```

Go into the project folder

```bash
cd Marketing-Campaign-Performance-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📌 Future Improvements

- Hyperparameter Optimization
- XGBoost Implementation
- LightGBM Integration
- Real-time API Deployment
- Cloud Deployment
- Explainable AI using SHAP

---

# 📚 Learning Outcomes

This project demonstrates:

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Regression
- Classification
- Model Evaluation
- Model Deployment
- Streamlit Development

---

# 👨‍💻 Author

**Puneeth Raja K**
