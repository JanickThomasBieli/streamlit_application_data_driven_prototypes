import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df=pd.read_csv("./Data_and_ML_model/Bank_Credit_Card_Customer_Churners.csv")

model = joblib.load("./Data_and_ML_model/best_perfoming_model_overall_XGBC.joblib")

# I will add the logo of the company to the sidebar
st.sidebar.image("./Data_and_ML_model/Logo_of_Bank.jpg", width=200)

# I will create three tabs, a welcome tab, one for displaying the data and the other for the model to make predictions
tabs = ["Welcome Page", "Customers Data Analysis", "Prediction Model"]
st.sidebar.markdown("# Please choose a page:")
page = st.sidebar.radio("Select a page:", tabs, label_visibility="collapsed")

if page == "Welcome Page":

    # Create two columns
    col1, col2 = st.columns(2)

    # Use the first column to display the title
    with col1:
        st.markdown("<h2 style='text-align: left; color: black;'>Identification of Fortunata Bank's Credit Card Customers that will likely churn in the future</h2>", unsafe_allow_html=True)
    
    # Use the second column to display the logo
    with col2:
        st.image("./Data_and_ML_model/Logo_of_Bank.jpg", width=200)

    st.markdown("<h4 style='text-align: left; color: black;'>Welcome to the Fortunata Bank's Credit Card Customer Churn Data Analysis and Prediction App!</h4>", unsafe_allow_html=True)
    st.markdown("""
    This application **provides value to its users in two ways:**

    1. **Data Analysis:** This application analyzes the characteristics of the current credit card customers of Fortunata Bank, by providing an in-depth analysis of the customer data. This adds value to the bank by providing insights into the customer base, which can be used to improve customer satisfaction and retention. By publishing this analysis as part of the Streamlit application at hand, it is made available to a wide range of employees within the bank, who can use the insights to improve their work.

    2. **Churn Prediction:** This app predicts the likelihood of a bank credit card customer churning based on their demographic, some financial and transactional data.

    **Main Goal:** By identifying customers who are likely to churn, Fortunata Bank can proactively engage with these customers, offering them incentives to stay, which can significantly reduce the cost and need of acquiring new customers.
    """)

if page == "Customers Data Analysis":

    # Add a title to the sidebar
    st.sidebar.title("Sections of Customer Data Analysis:")

    # Add links to different sections
    st.sidebar.markdown("[1. Dataset Overview](#dataset-overview)")
    st.sidebar.markdown("[2. Basic Dataset Analysis](#basic-dataset-analysis)")
    st.sidebar.markdown("[3. Summary Statistics Numerical Features/Variables](#summary-statistics)")
    st.sidebar.markdown("[4. Distribution of the Categorical Variables](#distribution-of-categorical-variables)")
    st.sidebar.markdown("[5. The Target Variable Attrition Flag and its Distribution](#distribution-of-the-target-variable)")
    st.sidebar.markdown("[6. Distribution of the Target Variable in Relation to the Categorical Features](#distribution-of-the-target-variable-in-relation-to-the-categorical-features)")
    st.sidebar.markdown("[7. Heatmap - Correlation between Numerical Features](#heatmap-correlation-between-numerical-features)")


    st.header("Fortunata Bank's Credit Card Customers - Customer Data Analysis")
    st.write("This dataset from Fortunata Bank contains information about credit card customers and whether they churned or not.")
    st.markdown(f"The dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns.**")

    
    # I will display the first 5 rows of the dataset
    st.markdown("<a name='dataset-overview'></a>", unsafe_allow_html=True)
    st.subheader("1. Dataset Overview")
    st.markdown("##### First 5 rows of the dataset. This shall give you a first overview of the data.")
    st.write(df.head())

    # I will display the basic dataset analysis
    st.markdown("<a name='basic-dataset-analysis'></a>", unsafe_allow_html=True)
    st.subheader("2. Basic Dataset Analysis")
    col_columns, col_shape, col_missing_values = st.columns(3)

    with col_columns:
        # I will display the columns of the dataset
        st.markdown("##### Columns of the dataset:")
        st.write(df.columns)

    with col_shape:
        # I will display the shape of the dataset
        st.markdown("##### Shape of the dataset:")
        st.info(f"Number of rows: {df.shape[0]}")
        st.info(f"Number of columns: {df.shape[1]}")

    with col_missing_values:
        # I will display the missing values of the dataset
        st.markdown("##### Number of missing values per column:")
        st.write(df.isnull().sum())

    # I will display the summary statistics for numeric columns in the dataset
    st.markdown("<a name='summary-statistics'></a>", unsafe_allow_html=True)
    st.subheader("3. Summary statistics for the numeric features/variables in the dataset:")
    st.write(df.describe())

    # I will display the summary statistics for categorical columns in the dataset
    # More precisely, I will display the distribution of categorical variables
    st.markdown("<a name='distribution-of-categorical-variables'></a>", unsafe_allow_html=True)
    st.subheader("4. Distribution of categorical variables/features in the dataset:")

    # List of categorical features/columns
    # Attrition Flag is the target variable and will be excluded because it is not a feature
    cat_features = [col for col in df.columns if df[col].dtype == 'object' and col != 'Attrition_Flag']

    for col in cat_features:
        col_no_underscore = col.replace('_', ' ') # I replace the underscore with a space for better readability
        st.markdown(f"##### Distribution of the categorical feature {col_no_underscore}")
        
        # Create a bar chart
        plt.figure(figsize=(10, 6))
        counts_ = df[col].value_counts()
        counts_.plot(kind='bar')
        plt.ylabel("Count")
        plt.xticks(rotation=60)

        # I add the counts above the bars
        for _i_, _value_ in enumerate(counts_):
            plt.text(_i_, _value_ + 50, str(_value_), ha = 'center')

        # Display the plot in Streamlit
        st.pyplot(plt)

    # I will display the distribution of the target variable in the dataset
    st.markdown("<a name='distribution-of-the-target-variable'></a>", unsafe_allow_html=True)
    st.subheader("5. The Target Variable Attrition Flag and its Distribution")
    st.markdown("**The column 'Attrition_Flag' contains the target variable**. It indicates whether a customer is likely to churn or not.")
    
    st.write("""Either the customer is likely to churn (Attrition Flag equal to "Attrited Customer", respectively to 1, the positive class) 
                 or not (Attrition Flag equal to "Existing Customer", respectievly to 0, the negative class).""")

    col_target_1, col_target_2 = st.columns(2)

    with col_target_1:
        # I will display the unique values in the 'Attrition_Flag' column
        st.markdown("##### Unique values in the 'Attrition_Flag' column")
        st.write(df["Attrition_Flag"].unique())

    with col_target_2:
        # I will display the value counts of the 'Attrition_Flag' column
        st.markdown("##### Value counts of the 'Attrition_Flag' column")
        st.write(df["Attrition_Flag"].value_counts())

    # I will visualize the distribution of the 'Attrition_Flag' column
    st.markdown("##### Visualization of the distribution of the target variable")
    plt.figure(figsize=(10, 6))
    counts = df["Attrition_Flag"].value_counts()
    counts.plot(kind='bar')
    plt.ylabel("Count")
    plt.xticks(rotation=60)

    # I add the counts above the bars
    for i, value in enumerate(counts):
        plt.text(i, value + 50, str(value), ha = 'center')

    st.pyplot(plt)

    st.markdown(f"""**Findings regarding the distribution of the target variable**: Having a look at the distribution of the target variable, we can see that the **dataset is imbalanced**. 
             There are more customers that are likely to stay with the bank (Existing Customers) than customers that are likely to churn (Attrited Customers).
             More precisely, there are **{df['Attrition_Flag'].value_counts()[0]/df.shape[0]:.2f}% of customers in the dataset that are likely to stay with the bank**, while
            there are **only {df['Attrition_Flag'].value_counts()[1]/df.shape[0]:.2f}% of customers that are likely to churn**.
            This imbalance in the dataset was taken into account when training the prediction model.""")
    
    # I will display the distribution of the target variable in relation to the categorical features
    st.markdown("<a name='distribution-of-the-target-variable-in-relation-to-the-categorical-features'></a>", unsafe_allow_html=True)
    st.subheader("6. Distribution of the Target Variable in Relation to the Categorical Features")
    st.markdown("**The following plots and tables show the distribution of the target variable in relation to the categorical features.**")

    for col in cat_features:
        col_no_underscore = col.replace('_', ' ') # I replace the underscore with a space for better readability
        st.markdown(f"##### Distribution of the target variable in relation to the feature {col_no_underscore}")
        
        # I create bar charts
        plt.figure(figsize=(10, 6))
        counts = df.groupby([col, 'Attrition_Flag']).size().unstack()
        counts.plot(kind='bar', stacked=True)
        plt.ylabel("Count")
        plt.xticks(rotation=60)

        # I calculate the percentage of attrited customers for each category
        percentages = (counts['Attrited Customer'] / counts.sum(axis=1) * 100).round(2)

        # I display the plots in Streamlit
        st.pyplot(plt)

        st.markdown(f"**Percentage of attrited customers for each category of the feature {col_no_underscore}**")
        # Create a DataFrame with the percentages and display it in Streamlit
        percentages_df = pd.DataFrame({f'{col} Categories': counts.index, 'Percentage of Attrited Customers [%]': percentages})
        # I sort the DataFrame in descending order according to the percentage of attrited customers
        percentages_df = percentages_df.sort_values(by='Percentage of Attrited Customers [%]', ascending=False)
        
        # I display the DataFrame in Streamlit without the index using HTML
        st.markdown(percentages_df.reset_index(drop=True).to_html(index=False), unsafe_allow_html=True)

    # In a last step in the data analysis part I will display a heatmap showing the correlation between the numerical features
    st.markdown("<a name='heatmap-correlation-between-numerical-features'></a>", unsafe_allow_html=True)
    st.subheader("7. Heatmap - Correlation between Numerical Features")

    st.markdown("##### The following heatmap shows the correlation between the numerical features in the dataset.")
    st.write("""The correlation coefficient ranges from -1 to 1. If the correlation is 
             close to 1, it means that there is a strong positive correlation between the two variables. 
             If the correlation is close to -1, it means that there is a strong negative correlation between 
             the two variables. Multicollinearity can be a problem when we train machine learning models, 
             especially in case of training linear models.""")
    
    plt.figure(figsize=(12, 8))
    # I select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    # I compute and plot the correlation matrix
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)
    
if page == "Prediction Model":
    tab_intro , tab_prediction, tab_performance = st.tabs(["Introduction of the Prediction Model", "Performing Predictions", "Model Performance"])

    with tab_intro:
        st.header("Introduction of the Prediction Model")
        st.subheader("Model to predict whether a Credit Card Customer is liekly to churn or not")
        st.markdown(""" 
                **Usage of the model:** This model predicts the likelihood of a customer churning based on their demographic and transactional data. The model will return, based on the input data, whether the customer the input data belongs to is likely to churn or not.
                Accordingly it is a binary classification model. Either the customer is likely to churn or not. The positive class is the 'Attrited Customer' class (customers likely to churn) and 
                the negative class is the 'Existing Customer' class (customers likely to stay with the company). 
                       
                **Model Traning Process:** The model was trained based on the Bank Churners dataset. Multiple models were trained (Random Forest Classifiers and XGBoost Classifiers), using multiple hyperparameters. 
                The best performing model was the XGBoost Classifier which was saved as a joblib file and is being used to make predictions in this app.
                """)

    with tab_prediction:
        # I will create a form for the user to input the data of the customer
        st.header("Performing Predictions")
        st.markdown("##### Explanation: Input the customer's data to get the prediction considering whether this specific customer will churn or not")
        customer_data = {}

        st.subheader("Demographic Infomration - Please input the following data of the customer:")

        col_1, col_2 = st.columns(2)

        with col_1:
            customer_data["Customer_Age"] = st.slider("Customer Age", min_value = 18, max_value= 120, value=int(df["Customer_Age"].mean()), step=1)
            customer_data["Dependent_count"] = st.slider("Number of dependents", min_value = 0, max_value= 12, value=int(df["Dependent_count"].mean()), step=1)
            customer_data["Card_Category"] = st.selectbox("Card Category", df["Card_Category"].unique())

        with col_2:
            customer_data["Gender"] = st.selectbox("Gender", df["Gender"].unique())
            customer_data["Education_Level"] = st.selectbox("Education Level", df["Education_Level"].unique())
            customer_data["Marital_Status"] = st.selectbox("Marital Status", df["Marital_Status"].unique())

        st.subheader("Finanacial and transactional information - Please input the following data of the customer:")

        col_3, col_4, col_5 = st.columns(3)

        with col_3:
            customer_data["Months_on_book"] = st.slider("Months on book", min_value=0, max_value=66, value=36, step=1)
            customer_data["Total_Relationship_Count"] = st.slider("Total relationship count", min_value=1, max_value=10, value=3, step=1)
            customer_data["Credit_Limit"] = st.slider("Credit limit", min_value=0, max_value=100000, value= 4444, step=1)
            customer_data["Total_Revolving_Bal"] = st.slider("Total revolving balance", min_value=0, max_value=50000, value= 10000, step=1)
            customer_data["Total_Trans_Amt"] = st.slider("Total transaction amount", min_value=0, max_value=10000, value= 1900, step=100)
           

        with col_4:
            customer_data["Income_Category"] = st.selectbox("Income Category", df["Income_Category"].unique())
            customer_data["Months_Inactive_12_mon"] = st.slider("Months inactive in the last 12 months", min_value=0, max_value=12, value=2, step=1)
            customer_data["Contacts_Count_12_mon"] = st.slider("Contacts in the last 12 months", min_value=0, max_value=14, value=1, step=1)
            customer_data["Avg_Open_To_Buy"] = st.slider("Average open to buy", min_value=0, max_value=100000, value= 66666, step=1)
        
        with col_5:
            customer_data["Total_Trans_Ct"] = st.slider("Total transaction count", min_value=0, max_value=150, value= 20, step=1)
            customer_data["Total_Ct_Chng_Q4_Q1"] = st.slider("Total count change Q4 to Q1", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            customer_data["Avg_Utilization_Ratio"] = st.slider("Average utilization ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            customer_data["Total_Amt_Chng_Q4_Q1"] = st.slider("Total amount change Q4 to Q1", min_value=0.0, max_value=5.0, value=2.6, step=0.01)

        st.subheader("""After you have provided all needed data from the customer, please click the 'Predict' button to get the prediction regrading whether the customer is likely to churn or not.""")
        st.markdown("The **default values** will return a prediction, predicting that a **customer having the respective data would be likely to churn**.")
        st.markdown("If you would like to receive **not likely to churn** as prediction, please change income catgeory to $80K - $120K, total transaction amount to 6200, total transaction count to 100, and total count change Q4 to Q1 to 0.5.")
        st.markdown("Of course, there are many other combinations that will lead to a prediction of **not likely to churn** or **likely to churn**. The settings above were just examples")
    
        # I will add a button to make the prediction
        if st.button('Predict'):
            # I will convert the customer data to a DataFrame
            customer_df = pd.DataFrame([customer_data])

            # I will make the prediction, using the XGB Classifier model that was imported above as a joblib file
            prediction = model.predict(customer_df)

            # I will display the prediction
            if prediction[0] == 0:
                st.success("The customer **is not likely to churn.** So, the bank can expect the customer to stay with the bank and no measurements need to be taken.")
            
            # if the prediction is 1
            else:
                st.error("The customer **is likely to churn.** So, the bank should take measurements to prevent the customer from leaving the bank (e.g. targeted marketing).")

    with tab_performance:
        st.header("Model Performance")
        st.subheader("The model performance of the XGBoost Classifier (best performing model of all the models that were trained) is displayed below.")
        st.write("""The positive class is the 'Attrited Customer' class (customers likely to churn) and the 
                 negative class is the 'Existing Customer' class (customers likely to stay with the company)""")
        
        # I display the hyperparameters of the best performing model
        st.markdown("#### Hyperparameters of the best performing model (XGBoost Classifier), found through grid search in the  previous training process:")
        
        # I extract the hyperparameters of the model used in this app
        hyperparameters = {
            'Learning Rate': model.named_steps['XGBClassifier'].learning_rate,
            'Max Depth': model.named_steps['XGBClassifier'].max_depth,  
            'Number of Estimators': model.named_steps['XGBClassifier'].n_estimators
            }

        # I convert the dictionary to a DataFrame
        hyperparameters_df = pd.DataFrame.from_dict(hyperparameters, orient='index', columns=['Values of the Hyperparameters'])

        # I format the DataFrame to display values with 2 decimal places
        hyperparameters_df = hyperparameters_df.style.format("{:.2f}")

        # Display the DataFrame as a table in Streamlit
        st.table(hyperparameters_df)

         # I will display the confusion matrix of the model
        st.markdown("#### Confusion Matrix on previously unseen data, respectively on the test data:")
        st.write("""The confusion matrix shows the number of True Positives, False Positives, True Negatives, and False Negatives. 
                 It is a good way to evaluate the performance of a binary classification model.""")
        st.write("""The confusion matrix can also be used to calculate other metrics such as Precision, Recall, and Accuracy.""")
        st.markdown("""**The confusion matrix of the model on previously unseen data (test data) looks as follows:**""")
        
        # I import the joblib-file containing the confusion matrix
        confusion_matrix = joblib.load("./Data_and_ML_model/confusion_matrix.joblib")

        # Create a DataFrame from the confusion matrix
        confusion_matrix = pd.DataFrame(confusion_matrix)

        # I rename the index and columns of the DataFrame so that the confusion matrix is displayed in a more readable way
        confusion_matrix.index = ['Actual Negative', 'Actual Positive']
        confusion_matrix.columns = ['Predicted Negative', 'Predicted Positive']


        # I display the confusion matrix in Streamlit
        st.table(confusion_matrix)

        # I will display the ROC-AUC score, Precision, Recall, and Accuracy of the model
        st.markdown("#### Model Metrics on previously unseen data, respectively on the test data:")
        # I create 2 columns
        col_ROC_AUC_Precision, col_Recall_Accuracy = st.columns(2)

        # I use the first column to display ROC_AUC and Precision
        with col_ROC_AUC_Precision:
            
            # I will display the ROC-AUC score of the model, using the st.expander() function
            with st.expander("ROC-AUC Score on previously unseen data (test data):"):
                st.write("0.9462")

            # I will display the Precision score of the model, using the st.expander() function
            with st.expander("Precision Score on previously unseen data (test data):"):
                st.write("0.9363")

        # I use the second column to display Recall and Accuracy
        with col_Recall_Accuracy:

            # I will display the Recall score of the model, using the st.expander() function
            with st.expander("Recall Score on previously unseen data (test data):"):
                st.write("0.9041")
 
            # I will display the Accuracy score of the model, using the st.expander() function
            with st.expander("Accuracy Score on previously unseen data (test data):"):
                st.write("0.9747")
        
        # I will display the ROC curve of the model
        st.markdown("#### ROC Curve of the model regarding its performance on previously unseen data (test data):")
        # I will display the ROC curve of the model by using the st.image() function and importing the image previosuly created and saved
        st.image("./Data_and_ML_model/ROC_AUC_curve.png")
