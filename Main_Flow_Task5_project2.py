import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load Reviews Dataset
reviews_df = pd.read_csv("reviews.csv")

# Preprocessing Function using NLTK
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(rf"[{string.punctuation}]", "", text)
    # Tokenize using TreebankWordTokenizer
    tokens = tokenizer.tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# Apply preprocessing
reviews_df['Cleaned Review'] = reviews_df['Review Text'].apply(preprocess_text)

# Encode sentiment labels
label_encoder = LabelEncoder()
reviews_df['Sentiment Label'] = label_encoder.fit_transform(reviews_df['Sentiment'])

# Text Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(reviews_df['Cleaned Review'])

# Convert to DataFrame for inspection (optional)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(tfidf_df.head())

# Display the cleaned dataset with labels
print(reviews_df[['Review Text', 'Cleaned Review', 'Sentiment', 'Sentiment Label']])

# Model Training using Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix,
    reviews_df['Sentiment Label'],
    test_size=0.2,
    random_state=42,
    stratify=reviews_df['Sentiment Label']
)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Model Evaluation
y_pred = logistic_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Deliverables
print("\nDeliverables:")
print("1. Preprocessed Dataset:")
print(reviews_df[['Review Text', 'Cleaned Review']].head())
print("\n2. Sentiment Classification Model:")
print("Logistic Regression model trained to classify sentiments.")
print("\n3. Evaluation Report:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\n4. Insights:")
reviews_df['Predicted Label'] = logistic_model.predict(tfidf_matrix)
reviews_df['Predicted Sentiment'] = label_encoder.inverse_transform(reviews_df['Predicted Label'])

# Examples of correct and incorrect predictions
correct = reviews_df[reviews_df['Sentiment Label'] == reviews_df['Predicted Label']]
incorrect = reviews_df[reviews_df['Sentiment Label'] != reviews_df['Predicted Label']]

print("\nExamples of correctly classified reviews:")
print(correct[['Review Text', 'Sentiment', 'Predicted Sentiment']].head())

print("\nExamples of incorrectly classified reviews:")
print(incorrect[['Review Text', 'Sentiment', 'Predicted Sentiment']].head())

# Common features of positive and negative reviews (optional simple word count)
pos_reviews = reviews_df[reviews_df['Sentiment'] == 'positive']['Cleaned Review'].str.split().explode().value_counts().head(10)
neg_reviews = reviews_df[reviews_df['Sentiment'] == 'negative']['Cleaned Review'].str.split().explode().value_counts().head(10)

print("\nTop words in positive reviews:")
print(pos_reviews)

print("\nTop words in negative reviews:")
print(neg_reviews)
