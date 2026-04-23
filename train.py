import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
import pickle

from preprocessing import clean_text

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("data/resume_data.csv")

# Normalize column names
df.columns = df.columns.str.lower().str.strip()

print("Columns:", df.columns)

# -------------------------------
# 2. Data Preprocessing
# -------------------------------
df['cleaned'] = df['resume_str'].apply(clean_text)

# Features & Labels
X = df['cleaned']
y = df['category']

# -------------------------------
# 3. Feature Engineering (TF-IDF)
# -------------------------------
tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),     # unigrams + bigrams
    stop_words='english',
    min_df=2,               # remove rare words
    max_df=0.8              # remove very common words
)

X = tfidf.fit_transform(X)

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. Model (Best for NLP 🔥)
# -------------------------------
model = LinearSVC(class_weight='balanced')

model.fit(X_train, y_train)

# -------------------------------
# 6. Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# 7. Save Model
# -------------------------------
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(tfidf, open("models/vectorizer.pkl", "wb"))

print("\nModel saved successfully!")