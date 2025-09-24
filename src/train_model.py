import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


# 🔹 Load and clean external dataset (UCI SMS Spam Collection)
def load_external_dataset(path):
    path = os.path.abspath(path)
    print(f"Loading dataset from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")

    # The UCI SMS dataset is TAB-separated
    df = pd.read_csv(
        path,
        encoding="latin-1",
        sep="\t",
        header=None,
        names=["label", "message"],
        on_bad_lines="skip"
    )

    print(f"  ▶ read_csv success with sep=\\t; shape={df.shape}")
    print("  ▶ Sample rows:\n", df.head())

    # Normalize labels
    df['label'] = df['label'].str.strip().str.lower()
    df = df[df['label'].isin(['ham', 'spam'])]

    # Map to 0/1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    print(f"✅ After cleaning: {df.shape}")
    print("🔎 Label distribution:\n", df['label'].value_counts())
    return df


# 🔹 Preprocessing function
def preprocess(text):
    text = text.lower()
    # Reduce elongated letters: freeeee -> free
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    # Remove URLs, numbers, punctuation
    text = re.sub(r'http\S+|www\S+|\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text


def main():
    # 1️⃣ Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "spam.csv")
    df = load_external_dataset(dataset_path)

    if df['label'].nunique() < 2:
        raise ValueError(
            f"Dataset does not contain both classes after cleaning. "
            f"Label unique values: {df['label'].unique()}"
        )

    # 2️⃣ Preprocess
    X = df['message'].astype(str).apply(preprocess)
    y = df['label']

    # 3️⃣ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4️⃣ Vectorize
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5️⃣ Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # 6️⃣ Evaluate
    y_pred = model.predict(X_test_vec)
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 7️⃣ Save model + vectorizer
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "spam_model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))
    print(f"\n✅ Model and vectorizer saved in {model_dir}/")


if __name__ == "__main__":
    main()
