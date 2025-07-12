

# 📘 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# 📂 2. Load Dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

df.head()

# 📊 3. Data Exploration
print("Dataset shape:", df.shape)
print("\nLabel counts:\n", df['label'].value_counts())
sns.countplot(data=df, x='label')

# 🧹 4. Preprocessing
# Convert labels to binary (ham = 0, spam = 1)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Split dataset
X = df['message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 🔠 5. Text Vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 🧠 6. Train Model using Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 🎯 7. Predict
y_pred = model.predict(X_test_tfidf)

# ✅ 8. Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 📝 9. Test with Custom Message
sample_msg = ["Congratulations! You've won a free ticket to Bahamas. Click now!"]
sample_vec = vectorizer.transform(sample_msg)
prediction = model.predict(sample_vec)
print("Prediction:", "Spam" if prediction[0] == 1 else "Ham")
