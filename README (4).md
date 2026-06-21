# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn (cancel their subscription), built with Python, Pandas, Scikit-learn, and Streamlit.

## About the Dataset

The dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) contains records of 7,000+ telecom customers, including demographics, account information (tenure, contract type), and the services they've subscribed to. The target variable indicates whether the customer churned.

## Project Workflow

1. **Data Cleaning** — Loaded and inspected the dataset, identified and fixed a data type issue in the `TotalCharges` column (it was stored as text due to blank values), and removed rows with missing data.

2. **Exploratory Data Analysis (EDA)** — Analyzed churn patterns across different customer segments. Two key findings stood out:
   - **Contract type** strongly affects churn: month-to-month customers churn at **42.7%**, compared to just **2.8%** for two-year contract customers.
   - **Tenure** matters: customers in their first few months show the highest churn risk, while long-tenured customers rarely churn.

3. **Feature Engineering** — Converted categorical variables (gender, contract type, services, etc.) into numerical format using one-hot encoding.

4. **Model Training** — Trained a Logistic Regression model to classify customers as likely to churn or not.

5. **Evaluation** — Achieved **79% accuracy**. Also examined the confusion matrix and classification report, which revealed that the model's recall for the churn class was lower (52%), meaning it misses some churners — an important limitation to address with techniques like class balancing in future iterations.

6. **Deployment** — Built an interactive web app using Streamlit, where users can input customer details and get a real-time churn prediction with probability score.

## Tools Used

- Python
- Pandas
- Seaborn & Matplotlib (visualization)
- Scikit-learn (machine learning)
- Streamlit (app deployment)
- Joblib (model persistence)

## How to Run This Project

```bash
pip install pandas scikit-learn streamlit joblib seaborn matplotlib
streamlit run app.py
```

## Files in This Repository

- `churn_analysis.ipynb` — Full analysis notebook (data cleaning, EDA, model training, evaluation)
- `app.py` — Streamlit web app for live predictions
- `churn_model.pkl` — Saved trained model
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — Dataset used

## Key Takeaway

Customers on month-to-month contracts and those in their early tenure are at the highest risk of churning. This insight could help a business prioritize retention efforts — for example, offering incentives to switch to longer contracts or extra support during a customer's first few months.
