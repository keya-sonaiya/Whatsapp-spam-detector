import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import pandas as pd
from src.data_preprocessing import load_chat
from src.Labelling import auto_label

def load_model():
    """
    Load trained model and vectorizer from models/ folder
    """
    model_path = os.path.join("models", "spam_model.pkl")
    vec_path = os.path.join("models", "vectorizer.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)

    return model, vectorizer

def clean_messages(df):
    """
    Ensure messages are strings and remove empty messages
    """
    df = df[df['message'].notnull()].copy()  # Remove None/NaN messages
    df['message'] = df['message'].astype(str).str.strip()
    df = df[df['message'] != ""].reset_index(drop=True)
    return df


def predict_chat(file_path):
    """
    Predict spam/ham for messages inside a WhatsApp chat file
    """
    # Load WhatsApp chat
    df = load_chat(file_path)
    df = clean_messages(df)

    if df.empty:
        raise ValueError("No messages loaded. Check your WhatsApp chat format.")

    # Auto-label obvious spam keywords
    df = auto_label(df)

    # Load trained model + vectorizer
    model, vectorizer = load_model()

    #  Predict messages that are NOT auto-labeled as spam
    df["prediction"] = "Spam"
    mask = ~df['auto_spam']
    if mask.any():  # Only if there are messages left to predict
        X_vec = vectorizer.transform(df.loc[mask, "message"])
        predictions = pd.Series(model.predict(X_vec), index=df.index[mask])
        df.loc[mask, "prediction"] = predictions.map({0: "Ham", 1: "Spam"})

    # Combine auto-labeled spam and model predictions
    df['final_prediction'] = df.apply(
        lambda row: "Spam" if row['auto_spam'] else row['prediction'], axis=1
    )

    return df


if __name__ == "__main__":
    test_file = os.path.join("data", "temp_chat.txt")
    results = predict_chat(test_file)

    print("Predictions on WhatsApp chat:")
    print(results[["sender", "message", "final_prediction"]].tail(20))  # last 20 messages
