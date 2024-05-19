## Fortunata Bank's Credit Card Customer Churn Prediction App

### Overview

The Fortunata Bank Credit Card Customer Churn Prediction App is designed to assist Fortunata Bank's 
employees in identifying customers who are likely to churn in the future. Utilizing data analytics 
and machine learning techniques, this application provides two main functionalities:

	1. Data Analysis: Explore in-depth analytics of current credit card customers to uncover 
	insights that could help enhance customer retention and satisfaction.

	2. Churn Prediction: Predict the likelihood of customers churning using demographic, 
	financial, and transactional data, enabling proactive engagement to improve retention. 
	These predictions are based on an XGBoost Classifier model. This model was initially trained 
	and selected among many other models for the course "Cloud Platforms - AWS" in term 2. 
	It was the best-performing model among the multiple models trained. This was not a big surprise 
	because XGBoost models often perform exceptionally well on tabular data like the data that we are 
	confronted with as part of the problem solved in this application.

### Reasons why XGBoost performs so well on tabular data:

	- Ensemble Learning: XGBoost is an implementation of gradient boosted decision trees designed 
	for speed and performance. Ensemble methods like XGBoost combine the predictions of several 
	base estimators (typically decision trees), which often leads to better generalization and 
	robustness than a single estimator. This ensemble method helps in reducing variance and bias, 
	making it effective for tabular data.

	- Regularization: XGBoost includes built-in L1 (Lasso) and L2 (Ridge) regularization which 
	prevents the model from overfitting.

	- Hyperparameter Tunability: XGBoost allows for fine-tuning a large number of parameters like 
	learning rate, depth of trees, and regularization terms, among others, which can be critically 
	adjusted to avoid overfitting and to optimize performance, making it highly adaptable to the dataset in question.

### How to Use the App

The application is divided into three main tabs:

	- Welcome Page
	- Customers Data Analysis
	- Prediction Model

#### Welcome Page

This tab introduces the app and its purposes, with a visual display of Fortunata Bank’s logo alongside.

#### Customers Data Analysis

Users can interact with various sections that provide analytics about the customer dataset, 
including the distribution of categorical variables, summary statistics, and more.

#### Prediction Model

This tab allows users to input customer data and receive predictions regarding potential churn. 
It also provides insights into the model's performance and other relevant statistics. 
Not least, this section of the application also dives into the analysis of which variables/features 
have the strongest predictive power when it comes to predicting credit card customer churn.

### Setup

There is a requirements.txt file provided to efficiently install all dependencies.

### File Structure

streamlit_app_Janick_Thomas_Bieli.py: Contains the main application code including the Streamlit interface.

Data_and_ML_model/: This directory contains the dataset, machine learning model, and other relevant data such 
as visualizations and pictures used in the application.

	- Bank_Credit_Card_Customer_Churners.csv: Dataset file.
	- best_performing_model_overall_XGBC.joblib: Pre-trained XGBoost model for predictions.
	- Logo_of_Bank.jpg: Logo displayed in the app.
	- confusion_matrix.joblib: Confusion matrix of the used XGBoost Classifier model.
	- ROC_AUC_curve.png: Visualization regarding how the used XGBoost Classifier model performs in terms of ROC-AUC.

### Dependencies

Python 3.8+
Streamlit: For creating the web application.
Pandas: For data manipulation.
Matplotlib & Seaborn: For data visualization.
Joblib: For loading the pre-trained model.
Numpy: For numerical operations.

### Additional Information

The application is designed to be intuitive and user-friendly, providing interactive visualizations to aid in 
understanding the data. Prediction outcomes help to better inform decision-making processes regarding customer 
retention strategies and measures that should be taken.
