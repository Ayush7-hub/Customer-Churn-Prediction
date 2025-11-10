

# 📊 Customer Churn Prediction using Machine Learning

## 📁 Project Overview

This project focuses on predicting **Customer Churn** in the telecom sector using **machine learning models**.
The goal is to identify which customers are likely to leave the service based on behavioral and demographic attributes, allowing the company to take proactive retention measures.

The project compares two models:

* **Decision Tree (CART)**
* **Logistic Regression**

After model comparison, **Logistic Regression** was selected for deployment due to its superior performance and interpretability.

---

## 🎯 Objectives

* Analyze telecom customer data to identify churn drivers.
* Perform **Exploratory Data Analysis (EDA)** and feature engineering.
* Build, train, and evaluate machine learning models.
* Compare model performance using **AUC-ROC**, **Accuracy**, and **F1 Score**.
* Deploy the best-performing model for real-time churn prediction.

---

## 🧩 Dataset Description

**Dataset Name:** Telco Customer Churn
**Records:** 7,043
**Attributes:** 21 (including categorical and numerical variables)

### Key Columns

| Feature                                                         | Description                                    |
| --------------------------------------------------------------- | ---------------------------------------------- |
| `customerID`                                                    | Unique ID for each customer                    |
| `gender`, `SeniorCitizen`, `Partner`, `Dependents`              | Demographic attributes                         |
| `tenure`                                                        | Number of months the customer has stayed       |
| `PhoneService`, `InternetService`, `TechSupport`, `StreamingTV` | Service-related attributes                     |
| `Contract`, `PaymentMethod`                                     | Account-related details                        |
| `MonthlyCharges`, `TotalCharges`                                | Billing information                            |
| `Churn`                                                         | Target variable (Yes = churned, No = retained) |

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was conducted using **Matplotlib** and **Seaborn** to visualize key churn trends.

### Major Insights:

* Customers with **month-to-month contracts** are most likely to churn.
* Higher **Monthly Charges** correlate strongly with churn.
* Lack of **Tech Support** and **Online Security** increases churn probability.
* **Electronic Check** as a payment method is most associated with churn.
* Long-term customers (**higher tenure**) are less likely to churn.

### Figures (Attach in Repository):

1. Churn Distribution Plot
2. Tenure vs. Churn (KDE Plot)
3. Contract Type vs. Churn (Bar Plot)
4. Internet Service vs. Churn (Bar Plot)
5. Payment Method vs. Churn
6. Tech Support vs. Churn
7. Correlation Heatmap

---

## 🧠 Model Development

### Models Used

| Model                | Type                    | Advantage                                        |
| -------------------- | ----------------------- | ------------------------------------------------ |
| Decision Tree (CART) | Non-linear / Rule-based | High interpretability, good for categorical data |
| Logistic Regression  | Linear / Probabilistic  | Good generalization, interpretable coefficients  |

### Preprocessing Steps

* Handled missing values (`TotalCharges`).
* Dropped non-predictive ID columns.
* Applied **Label Encoding** / **One-Hot Encoding** to categorical data.
* Scaled numerical features using **StandardScaler**.
* Used **Train-Test Split (80–20)** for validation.

---

## 📈 Model Evaluation

### Performance Comparison

| Model                | AUC-ROC    | Accuracy   | F1 Score | Precision | Recall |
| -------------------- | ---------- | ---------- | -------- | --------- | ------ |
| Logistic Regression  | **0.8105** | **0.7676** | 0.743    | 0.760     | 0.726  |
| Decision Tree (CART) | 0.8030     | 0.7669     | 0.739    | 0.751     | 0.724  |

**Key Takeaways:**

* Logistic Regression performed better in all major metrics.
* Decision Tree slightly overfitted, while Logistic Regression generalized well.
* Logistic Regression was finalized for deployment.

---

## ⚙️ Technologies and Tools Used

* **Language:** Python 3.10
* **Libraries:**

  * `pandas`, `numpy` — data processing
  * `matplotlib`, `seaborn` — visualization
  * `scikit-learn` — model building, evaluation, preprocessing
  * `joblib` — model serialization
* **Environment:** Jupyter Notebook
* **Version Control:** GitHub

---

## 🚀 Model Deployment

* The final **Logistic Regression model** was saved using `joblib.dump()`.
* It can be loaded into any Python environment or Flask/Django backend for real-time churn probability prediction.
* Future versions can include a **REST API** or **Streamlit web app** interface for interactive use.

---

## 📊 Key Results

* **Final Model:** Logistic Regression
* **AUC-ROC:** 0.8105
* **Accuracy:** 0.7676
* **Key Drivers of Churn:** Contract type, Tenure, Monthly Charges, Tech Support, Payment Method

**Business Insight:**
Customers on month-to-month plans with high monthly charges and no tech support are most likely to churn.
Offering loyalty rewards, technical assistance, and auto-payment discounts can reduce churn significantly.

---

## 🧾 Project Workflow Summary

```
1. Data Loading
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Encoding and Scaling
5. Model Building (Decision Tree, Logistic Regression)
6. Model Evaluation and Comparison
7. Model Deployment using Joblib
```

---

## 🧰 How to Run the Project

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook:**

   ```bash
   jupyter notebook pro.ipynb
   ```

4. **To use the trained model:**

   ```python
   import joblib
   model = joblib.load("churn_model.joblib")
   prediction = model.predict(new_data)
   ```

---

## 🧑‍💻 Author

**Name:** Ayush Lavania
**Institution:** SRM Institute of Science and Technology
**Department:** Computational Intelligence
**Course:** Inferential Statistics and Predictive Analytics (21AIC401T)
**Year:** 2025

---
