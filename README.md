# Employee Attrition Prediction Project

A comprehensive data science project that analyzes employee attrition patterns and builds a machine learning model to predict which employees are likely to leave the organization. This project uses advanced data imputation techniques, exploratory data analysis, and machine learning to provide actionable insights for HR departments.

## 📊 Project Overview

Employee attrition is a critical challenge for organizations, leading to increased costs, loss of institutional knowledge, and decreased productivity. This project aims to:

- Analyze employee data to identify key factors contributing to attrition
- Build a predictive model to identify employees at risk of leaving
- Provide insights to help organizations improve employee retention strategies

## ✨ Key Features

- **Advanced Data Imputation**: Multiple imputation strategies including group-based median/mode filling and predictive imputation using Linear Regression
- **Comprehensive Data Visualization**: Distribution analysis of numerical and categorical features
- **Feature Engineering**: One-hot encoding for categorical variables and standardization of numerical features
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique) to address imbalanced dataset
- **Machine Learning Model**: Logistic Regression classifier with performance evaluation metrics
- **Feature Importance Analysis**: Identification of key factors influencing employee attrition

## 📁 Dataset

The project uses an employee attrition dataset (`employee_attrition_dataset.csv`) containing the following features:

### Employee Demographics
- **Age**: Employee age
- **Gender**: Male/Female
- **MaritalStatus**: Single/Married/Divorced

### Job-Related Information
- **JobRole**: Employee's job position
- **Department**: Work department
- **YearsAtCompany**: Tenure in years
- **PerformanceRating**: Performance evaluation score
- **MonthlyIncome**: Monthly salary

### Work Environment
- **BusinessTravel**: Travel frequency
- **DistanceFromHome**: Commute distance
- **EnvironmentSatisfaction**: Satisfaction with work environment
- **WorkLifeBalance**: Work-life balance rating

### Other Factors
- **OverTime**: Whether employee works overtime
- **NumCompaniesWorked**: Number of previous employers
- **PercentSalaryHike**: Percentage salary increase
- **Attrition**: Target variable (Yes/No)

## 🚀 Installation and Setup

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

Or install all dependencies at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### Running the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/aanya70/Employees-Project.git
   cd Employees-Project
   ```

2. **Ensure the dataset is available**
   - Place `employee_attrition_dataset.csv` in the project root directory

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook Employees_project.ipynb
   ```

4. **Run all cells** to reproduce the analysis and results

## 📊 Project Workflow

### 1. Data Loading and Initial Exploration
- Load the employee attrition dataset
- Identify missing values in the dataset

### 2. Data Cleaning and Imputation
- **Group-based imputation** for MonthlyIncome and EnvironmentSatisfaction using job role
- **Predictive imputation** for NumCompaniesWorked and DistanceFromHome using Linear Regression
- Validation of imputation completeness

### 3. Exploratory Data Analysis (EDA)
- Visualization of numerical feature distributions
- Analysis of categorical variables
- Correlation analysis between features

### 4. Feature Engineering
- One-hot encoding for categorical variables
- Feature standardization using StandardScaler
- Preparation of feature matrix (X) and target variable (y)

### 5. Handling Class Imbalance
- Application of SMOTE to balance the dataset
- Resampling to ensure equal representation of both classes

### 6. Model Training and Evaluation
- Train-test split (80-20 ratio)
- Logistic Regression model training
- Performance evaluation using:
  - Classification Report (Precision, Recall, F1-Score)
  - Confusion Matrix
  - ROC AUC Score

### 7. Feature Importance Analysis
- Identification of top features contributing to attrition prediction
- Coefficient analysis from the trained model

## 🎯 Results

The Logistic Regression model achieves:

- **Overall Accuracy**: ~73%
- **ROC AUC Score**: ~0.81
- **Balanced Performance**: Similar precision and recall for both classes

### Top Predictive Features

Based on the model's coefficient analysis, the most influential factors for employee attrition include:

1. **Job Role** (Manager, Research Scientist, Laboratory Technician)
2. **Marital Status** (Married vs. Single)
3. **Department** (Research & Development, Sales)
4. **Business Travel Frequency**
5. **Gender**

## 🛠️ Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Imbalanced-learn**: SMOTE implementation for class imbalance
- **Jupyter Notebook**: Interactive development environment

## 📈 Model Performance Metrics

```
              precision    recall  f1-score   support

       False       0.73      0.75      0.74       165
        True       0.74      0.71      0.72       160

    accuracy                           0.73       325
   macro avg       0.73      0.73      0.73       325
weighted avg       0.73      0.73      0.73       325

ROC AUC Score: 0.8071
```

## 💡 Key Insights

1. **Job Role Impact**: Employees in managerial and scientific roles show different attrition patterns
2. **Work-Life Balance**: Significant factor in employee retention
3. **Travel Requirements**: Frequent business travel correlates with attrition
4. **Compensation**: Monthly income and salary hikes influence retention decisions
5. **Environment Satisfaction**: Lower satisfaction scores are associated with higher attrition risk

## 🔮 Future Enhancements

- [ ] Experiment with other ML algorithms (Random Forest, XGBoost, Neural Networks)
- [ ] Implement hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- [ ] Add time-series analysis for temporal attrition patterns
- [ ] Create an interactive dashboard for HR departments
- [ ] Deploy the model as a web application
- [ ] Implement SHAP values for better model interpretability

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Authors

- **aanya70** - *Initial work* - [GitHub Profile](https://github.com/aanya70)

## 🙏 Acknowledgments

- Thanks to the open-source community for the excellent libraries used in this project
- Dataset source: Employee Attrition Dataset
- Inspiration from HR analytics and workforce planning research

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue in the GitHub repository.

---

**Note**: This project is for educational and research purposes. Always ensure compliance with data privacy regulations when working with employee data in production environments.