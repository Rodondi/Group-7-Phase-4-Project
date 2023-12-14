# SENTIMENT CLASSIFICATION OF TWEETS ABOUT APPLE AND GOOGLE PRODUCTS

## Overview

This project involves the application of Natural Language Processing (NLP) techniques to analyze sentiments expressed in tweets related to Apple and Google. The goal is to build and evaluate multiple machine learning models for sentiment classification.

## Project Structure

The project is organized as follows:

- **Data Collection:** Initial dataset obtained from [source].
- **Data Preprocessing:** Cleaning, tokenization, stopword removal, and lemmatization of tweets.
- **Model Training:** Building and tuning various machine learning models for sentiment analysis.
- **Evaluation:** Comprehensive evaluation of models based on metrics such as accuracy, precision, recall, and training time.
- **Model Selection:** Choosing the optimal model based on a balance between accuracy and efficiency.

## Data Preprocessing

The preprocessing steps include:
- Tokenization: Splitting tweets into individual words (tokens).
- Stopword Removal: Removing common words that do not contribute to sentiment.
- Lemmatization: Reducing words to their base or root form.

## Machine Learning Models

The following models were trained and evaluated:

1. **Baseline Model (Multinomial Naive Bayes):** A starting point for comparison.
2. **Tuned Logistic Regression:** Optimized logistic regression with hyperparameter tuning.
3. **Tuned Support Vector Machines (SVM):** Optimized SVM with hyperparameter tuning.
4. **Tuned Random Forest:** Optimized random forest with hyperparameter tuning.

Models were trained on different preprocessed versions of the data, including no preprocessing, stemming, and lemmatization.

## Evaluation Metrics

Models were evaluated using the following metrics:
- **Test Accuracy:** Percentage of correctly classified instances in the test set.
- **Precision:** Proportion of true positive predictions among all positive predictions.
- **Recall:** Proportion of true positive predictions among all actual positive instances.
- **Training Time:** Time taken to train each model.

## Model Selection

The optimized Support Vector Machines (SVM) model, trained on lemmatized tweets without stopwords ("Tuned SVM No SW, Lem"), emerged as the best performer, achieving a 72.9% accuracy. This model was selected based on a balance between accuracy and training efficiency.

## Conclusion

The analysis revealed diverse sentiments expressed in tweets related to Apple and Google. The "Tuned SVM No SW, Lem" model demonstrated superior performance and efficiency, making it the optimal choice for sentiment analysis tasks.