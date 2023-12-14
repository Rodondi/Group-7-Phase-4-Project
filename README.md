# From Churn to Retention: SyriaTel's Strategy for Success

## Overview

SyriaTel is a leading telecommunications company providing services to a diverse customer base. 
In recent years, SyriaTel has observed a concerning trend of customers discontinuing their services, which negatively impacts the company's revenue and profitability. 
Customer churn, the rate at which customers discontinue their subscriptions, has become a critical concern for SyriaTel.
This project aims to address the issue of customer churn for SyriaTel, a telecommunications company, by solving a binary classification problem. 
The primary goal is to determine which customers are likely to stay with the company and who is more likely to terminate their business with SyriaTel. 
Additionally, the project provides actionable insights and suggestions for the company based on the analysis of key features to reduce customer churn.

## Business and Data Understanding

### Stakeholders
The primary stakeholder for this project is SyriaTel's management team, while secondary stakeholders include the marketing department, and customer service team. 
This project aims to empower these stakeholders with data-driven insights to mitigate churn and enhance customer satisfaction.

### Business Problem
1. **Customer Identification**: This project aims to predict which customers are likely to stay with SyriaTel and which are likely to churn based on data analysis and classification models.
2. **Identifying Patterns**: We are uncovering patterns and insights that can help SyriaTel better understand its customers.
2. **Retention Strategies**: This Project also offers suggestions on how SyriaTel can improve customer retention by addressing specific aspects identified through data analysis.

### The Data
The dataset used for this project was obtained from https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-datasetis labeled "bigml_59c28831336c6604c800002a.csv."
The dataset contains various features related to customer behavior and interactions with SyriaTel, including demographics, service plans, call records, demographics, usage patterns, and churn status. 
It serves as the foundation for building a predictive model and identifying key predictors of churn.

## Modeling

### Data Preparation
1. Data Loading: The project begins with loading the dataset, consisting of 20 features and one target variable (churn), with no missing data.
2. Data Preprocessing: Several data preprocessing steps are performed, such as renaming columns, handling binary categorical features, and converting data types.
3. Feature Examination: A thorough examination of the features is conducted, providing insights into the relationships between features and customer churn.
   
### Classification Models
1. **Baseline Model**: A baseline model is built using a Decision Tree classifier, and its performance metrics are assessed. This model provides a starting point for further analysis.
2. **Model Comparisons**: Various classification models, including Decision Tree, K-Nearest Neighbors, Random Forest, and Gradient Boosting, are compared to select the best-performing model.
3. **Optimization**: The Gradient Boosting Classifier is selected as the final model and optimized using grid search to achieve the best hyperparameters.
4. **Final Model**: The optimized Gradient Boosting Classifier is the chosen model for the project, and its performance metrics are evaluated.

## Evaluation

### Model Performance
The final model, a Gradient Boosting Classifier, demonstrates excellent performance on both training and test data, with high accuracy, recall, and F1 scores. The model's feature importance analysis reveals that the most critical factors affecting customer churn are customer service calls, international plans, and total day minutes.

## Conclusion

From the classification models and feature analysis, several key insights and recommendations for SyriaTel emerge:

1. **Customer Service Calls**: Customers with a high number of service calls (more than 4 calls) have a significantly higher churn rate. The company should optimize customer service interactions and implement proactive strategies to address customer issues effectively.

2. **International Plan**: Customers with international plans are more likely to churn. The company should assess the value and satisfaction of international plans and consider offering enhanced benefits to reduce churn.

3. **Total Day Minutes**: The churn rate is highest for customers with total day minutes greater than 315.6 and lowest for those with less than 46.5 minutes. The company should focus on retaining customers with day minutes between these thresholds.

4. **Geographical Insights**: Certain states, such as AZ, AK, and HI, have lower churn rates, indicating the presence of unique customer behaviors or market conditions. The company should explore these states for successful strategies to improve overall customer retention.

By implementing these recommendations, SyriaTel can take proactive steps to reduce customer churn, improve customer satisfaction, and ultimately enhance its bottom line. Continuous adaptation and data-driven decision-making will be essential for long-term success in a highly competitive telecommunications market.