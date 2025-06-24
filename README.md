# spam-email-classifier
A beginner-friendly machine learning project that classifies text messages as spam or not spam (ham) using Natural Language Processing (NLP) techniques and a Naive Bayes classifier.

Accuracy: 0.97

Classification Report:

              precision    recall  f1-score   support

         ham       0.98      1.00      0.99       965
        spam       0.98      0.88      0.93       150


# Problem Statement
Spam messages are a common nuisance in email and messaging platforms. This project builds a simple spam classifier using Python, helping detect and filter out unwanted messages based on their content.

# Dataset
[UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
Contains 5,574 messages labeled as either `spam` or `ham`

# Technologies used:
-Python
- Pandas for data manipulation
- Scikit-learn for machine learning
- TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization
- Naive Bayes Classifier for classification
- NLP for preprocessing

# Working:
1. Loads and cleans text data
2. Converts text to numerical features using **TF-IDF**
3. Trains a **Multinomial Naive Bayes model**
4. Evaluates the model using accuracy and classification report
5. Allows predictions on new, custom text inputs

# To run:
git clone https://github.com/yourusername/spam-email-classifier.git
cd spam-email-classifier

This project is for educational purpose only

# Author
Aditya Deshpande
LinkedIn: www.linkedin.com/in/aditya-deshpande-5b120b305
