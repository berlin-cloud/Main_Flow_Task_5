# 🧠 Main Flow Task 5 - Logistic Regression Projects

This repository contains two mini-projects submitted as part of the **Data Science with Python Internship** at **Main Flow Services and Technologies Pvt. Ltd.** Both projects apply **Logistic Regression** to solve real-world classification problems.

---

## 📁 Projects

### 🔹 Project 1: Student Performance Classification

**Objective**: Predict whether a student will pass or fail based on two features:
- Study Hours
- Attendance

**Technologies Used**:
- Pandas, Matplotlib, Seaborn
- Scikit-learn (LogisticRegression, train_test_split, confusion_matrix)

**Key Highlights**:
- Exploratory Data Analysis (EDA) with visualizations
- Outlier detection with boxplots
- Logistic Regression model to classify pass/fail
- Evaluation using Accuracy and Confusion Matrix

**Insights**:
- Students with more study hours and higher attendance are more likely to pass.
- Both features are positively correlated with performance.

---

### 🔹 Project 2: Sentiment Analysis of Reviews

**Objective**: Classify customer reviews as **positive** or **negative** using Natural Language Processing (NLP).

**Technologies Used**:
- Pandas, Numpy, Matplotlib, Seaborn
- NLTK for preprocessing (stopwords, lemmatization, tokenization)
- TF-IDF Vectorizer
- Scikit-learn (LogisticRegression, LabelEncoder, classification metrics)

**NLP Pipeline**:
- Text cleaning (punctuation & case removal)
- Tokenization using `TreebankWordTokenizer`
- Stopword removal and lemmatization
- TF-IDF vectorization for feature extraction

**Evaluation Metrics**:
- Accuracy, Precision, Recall, F1-Score
- Classification Report
- Visualization of correctly vs incorrectly classified reviews

**Insights**:
- Common keywords identified for positive and negative reviews
- Model performs well in capturing sentiment based on word usage

---

## 📌 Deliverables

- ✅ Cleaned and preprocessed datasets
- ✅ Well-commented code for reproducibility
- ✅ Evaluation reports for both models
- ✅ Key insights and feature importance analysis

---

## 🔗 Author

**S. Berlin Samvel Pandian**  
🎓 Artificial Intelligence & Machine Learning  
🔗 [LinkedIn](https://www.linkedin.com/in/s-berlin-samvel-pandian007)

---

## 📂 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/berlin-cloud/MainFlowTask5.git
   cd MainFlowTask5
