# ===============================
# 0️⃣ Imports
# ===============================
import os
import pandas as pd
import re, string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ===============================
# Set current working directory to script folder
# ===============================
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Current working directory:", os.getcwd())

# ===============================
# 1️⃣ Create Tiny Sample Dataset
# ===============================
data = {
    "Consumer complaint narrative": [
        "I was charged incorrectly by my bank",
        "Debt collector called me multiple times",
        "Loan application was rejected unfairly",
        "Mortgage payment dispute with lender",
        "Credit card interest charged wrongly",
        "Debt collection agency harassed me",
        "Consumer loan process took too long",
        "Mortgage account was mismanaged"
    ],
    "Product": [
        "Credit reporting, repair, or other",
        "Debt collection",
        "Consumer Loan",
        "Mortgage",
        "Credit reporting, repair, or other",
        "Debt collection",
        "Consumer Loan",
        "Mortgage"
    ]
}

df = pd.DataFrame(data)
df.to_csv("consumer_complaints_sample.csv", index=False)
print("Sample CSV created: consumer_complaints_sample.csv")
print(df)

# ===============================
# 2️⃣ Preprocess Text
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['Consumer complaint narrative'].apply(clean_text)

# ===============================
# 3️⃣ Encode Labels
# ===============================
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['Product'])
print("\nLabel encoding mapping:")
for i, cls in enumerate(le.classes_):
    print(f"{i} -> {cls}")

# ===============================
# 4️⃣ Train/Test Split
# ===============================
# For tiny dataset, use 50% test size so each class is represented
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label_enc'], test_size=0.5, random_state=42, stratify=df['label_enc']
)

# ===============================
# 5️⃣ TF-IDF Vectorization
# ===============================
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ===============================
# 6️⃣ Train Model
# ===============================
model = LogisticRegression(max_iter=500, multi_class='auto', solver='lbfgs')
model.fit(X_train_tfidf, y_train)

# ===============================
# 7️⃣ Evaluation
# ===============================
y_pred = model.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.show()

# ===============================
# 8️⃣ Save Predictions
# ===============================
df_test = pd.DataFrame({
    'text': X_test,
    'true_label': le.inverse_transform(y_test),
    'predicted_label': le.inverse_transform(y_pred)
})
df_test.to_csv("consumer_complaints_predictions.csv", index=False)
print("\nPredictions saved as consumer_complaints_predictions.csv")

# ===============================
# 9️⃣ Save Model & Vectorizer
# ===============================
joblib.dump(model, "consumer_complaint_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Model, TF-IDF vectorizer, and label encoder saved.")
