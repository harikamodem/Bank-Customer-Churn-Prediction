
# Customer Churn Prediction Analysis

##  Goal
The primary objective of this project is to analyze customer data from a bank and build machine learning models to predict customer churn (`Exited` column). The process involves comprehensive data cleaning, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, and performance evaluation.

---

##  Data Preprocessing & Cleaning

### 1. Data Loading and Missing Values
* The `Churn.csv` dataset is loaded.
* Missing values are identified in the `Geography` (1), `Age` (1), `HasCrCard` (1), and `IsActiveMember` (1) columns.
* **Simple Imputation:** All rows with any missing values are dropped using `data.dropna()`.

### 2. Feature Engineering
* **Irrelevant Columns Dropped:** `RowNumber`, `CustomerId`, and `Surname` are removed as they are non-predictive identifiers.
* **Categorical Encoding:**
    * `Gender` and `Geography` are converted into numerical form using **Label Encoding**.

### 3. Handling Class Imbalance
* The original target class distribution showed a significant imbalance (`Exited=0`: 7960, `Exited=1`: 2038).
* **SMOTE (Synthetic Minority Over-sampling Technique)** is applied to balance the dataset, resulting in an equal count for both classes (`Exited=0`: 7960, `Exited=1`: 7960).

### 4. Outlier Treatment & Scaling
* **Outlier Removal:** Outliers in `Age`, `NumOfProducts`, and `CreditScore` are filtered using the **Interquartile Range (IQR)** method ($Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$).
* **Feature Scaling:** Continuous features (`CreditScore`, `Balance`, `EstimatedSalary`) are scaled using **MinMaxScaler** to normalize their range between 0 and 1.

---

##  Exploratory Data Analysis (EDA)

Key observations from the visualizations:

* **Geography:** France has the highest overall number of customers, but Germany shows a higher proportion of churn.
* **Gender:** Female customers show a higher **rate** of churn compared to male customers.
* **NumOfProducts:** Customers with 3 or 4 products have a very high tendency to churn (almost all of them).
* **Age:** Customers who churn (`Exited=1`) tend to be older, as shown by the box plot where the median age is higher for the churned group.
* **Correlation:** The heatmap is used to check for multicollinearity among features.

---

##  Model Training & Evaluation

The balanced and scaled data is split into 80% training and 20% testing sets (`random_state=42`).

### 1. Baseline Model Performance (Before Hyperparameter Tuning)

| Model | Accuracy Score |
| :--- | :--- |
| Random Forest | **0.8829** |
| Decision Tree | 0.8251 |
| K-Neighbors | 0.7836 |
| Logistic Regression | 0.7560 |

### 2. Hyperparameter Tuning (GridSearchCV)

Hyperparameter tuning was performed on all models to find the optimal configuration:

| Model | Best Parameters |
| :--- | :--- |
| **Random Forest** | `{'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 150}` |
| **Decision Tree** | `{'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 10}` |
| **K-Neighbors** | `{'n_neighbors': 9, 'weights': 'distance'}` |
| **Logistic Regression** | `{'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}` |

### 3. Final Model Performance

| Model | Accuracy Score | R-Squared Score |
| :--- | :--- | :--- |
| **Random Forest** | **0.8851** | **0.5400** |
| Decision Tree | 0.8401 | 0.3603 |
| K-Neighbors | 0.7864 | 0.1454 |
| Logistic Regression | 0.7563 | 0.0248 |

### 4. Conclusion

The **Random Forest Classifier** is the best performing model for this customer churn prediction task, achieving the highest accuracy of **88.51%** and the highest $R^2$ score (which, while typically used for regression, confirms the superior fit) after hyperparameter tuning. The Confusion Matrix plots visually confirm its better performance in classifying both non-churners (0) and churners (1).

---
