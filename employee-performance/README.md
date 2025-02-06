# Employee Performance Analysis and Prediction

## 1. Introduction

### Overview
This project aims to analyze and predict employee performance scores using a dataset containing key information about employees.

### Objectives
- Preprocess and explore the dataset.
- Build a robust predictive model for performance scores.
- Deploy the model as an interactive Streamlit app.

## 2. Data Preprocessing

### Steps:
- Load the dataset.
- Handle missing values.
- Encode categorical variables.
- Normalize numerical features.
- Detect and treat outliers.

### Dataset Columns (Selected Features):
- **Department**: The department the employee belongs to.
- **Gender**: Male, Female, or Other.
- **Age**: Employee's age.
- **Job_Title**: Employee's role within the company.
- **Years_At_Company**: Number of years the employee has worked at the company.
- **Education_Level**: High School, Bachelor, Master, or PhD.
- **Performance_Score**: The performance rating of the employee.
- **Monthly_Salary**: The salary of the employee.
- **Projects_Handled**: Number of projects managed by the employee.
- **Overtime_Hours**: Number of overtime hours worked.
- **Promotions**: Number of promotions received.
- **Employee_Satisfaction_Score**: A score indicating employee satisfaction.

### Preprocessing Steps:
- Dropped irrelevant columns: Employee_ID, Hire_Date, Work_Hours_Per_Week, Sick_Days, Remote_Work_Frequency, Team_Size, Training_Hours, and Resigned.
- Checked and handled missing values.
- Removed duplicated records.
- Normalized numerical features where necessary.

## 3. Exploratory Data Analysis (EDA)

### Key Insights:
- **Salary Analysis:** Average salary distribution by department.
- **Performance Score Distribution:** Visualizing employee performance ratings.
- **Gender Distribution:** Breakdown of employees by gender.
- **Correlation Analysis:** Understanding relationships between different features.

### Sample Visualization:
```python
import matplotlib.pyplot as plt
import seaborn as sns

data.groupby("Department")["Monthly_Salary"].mean().sort_values(ascending=True).plot(kind="barh", color="skyblue")
plt.title("Average Salary by Department")
plt.xlabel("Average Salary")
plt.ylabel("Department")
plt.show()
```

## 4. Model Development

### Steps:
- Split data into training and testing sets.
- Train multiple machine learning models:
  - Logistic Regression
  - Random Forest
- Evaluate models using accuracy, confusion matrix, and classification report.

## 5. Deployment
- The trained model will be deployed using **Streamlit** for interactive predictions.
- Users can input employee details and receive a predicted performance score.

## 6. Conclusion
This project provides HR teams with data-driven insights into employee performance trends, helping improve decision-making regarding promotions, salary adjustments, and workforce planning.

---

## Project Dictionary

| Term                      | Definition |
|---------------------------|------------|
| **EDA**                   | Exploratory Data Analysis - examining and visualizing data to uncover insights. |
| **Normalization**         | Scaling numerical features to a standard range. |
| **Categorical Encoding**  | Converting categorical variables into a numerical format for machine learning models. |
| **Performance Score**     | A rating system representing employee productivity and efficiency. |
| **Feature Engineering**   | The process of selecting and transforming variables to improve model performance. |
| **Machine Learning Model** | An algorithm trained on historical data to make predictions. |
| **Confusion Matrix**      | A table used to evaluate classification model performance. |
| **Streamlit**             | A Python library for creating interactive web apps. |
| **Outlier Detection**     | Identifying and handling abnormal data points that can affect model accuracy. |

---
