import pandas as pd
import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Loading and exploring dataset
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
file_path=os.path.join(BASE_DIR, "spam.csv")
df= pd.read_csv(file_path, encoding='latin-1')[['v1','v2']]
df.columns=['label','text']
df.head()

# Preprocessing data
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

df['clean_text'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words='english')
x = vectorizer.fit_transform(df['clean_text'])

# Encode labels
le=LabelEncoder()
y=le.fit_transform(df['label']) 

# Train model
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model=MultinomialNB()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred)*100)

# Saving the model
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Both model saved successfully")

# Testing 
sample = ["You've won a $1000 Walmart gift card. Click to claim now!"]
sample_clean = [clean_text(sample[0])]
sample_vector = vectorizer.transform(sample_clean)

prediction = model.predict(sample_vector)
print("Prediction:", le.inverse_transform(prediction)) 