# WhatsApp Spam Detector

A machine learning–powered web application built with **Streamlit** that identifies spam messages from exported WhatsApp chats.

This platform uses a combination of **Natural Language Processing (NLP)**, **heuristic auto-labeling**, and a **Multinomial Naive Bayes** classifier to categorize messages as spam or ham. Additionally, it provides interactive, data-rich insights into your WhatsApp conversation—such as message trends, sender activity, top emojis, and word clouds—all wrapped in a clean, interactive GUI.

---

## Dataset

This project utilizes the **[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)** dataset (or similarly formatted data) to train its machine learning model. You can also find it on [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
- **Location:** The training file should be located at `data/spam.csv`.
- **Format:** The training script (`src/train_model.py`) expects a tab-separated (`\t`) structure, consisting of two columns: `label` (`ham` or `spam`) and `message`.
- **Preprocessing:** The training pipeline applies preprocessing such as lowercasing, stop-word removal, and character reduction mapping prior to text vectorization via TF-IDF.

---

## Dataset

This project utilizes the **[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)** dataset to train its machine learning model. You can also find it on [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
- **Location:** The training file should be located at `data/spam.csv`.
- **Format:** The training script (`src/train_model.py`) expects a tab-separated (`\t`) structure, consisting of two columns: `label` (`ham` or `spam`) and `message`.
- **Preprocessing:** The training pipeline applies preprocessing such as lowercasing, stop-word removal, and character reduction mapping prior to text vectorization via TF-IDF.

---

## ✨ Features

- **Upload & Analyze**: Drag and drop your exported WhatsApp chat (`.txt` without media).
- **Two-Layer Spam Detection**:
  - *Heuristic Auto-labeling*: Quickly traps absolute spam signatures (e.g., lottery, links, "click here").
  - *ML Classification*: Evaluates the remaining messages using a trained Naive Bayes classifier.
- **Deep Chat Analytics**:
  - Active sender statistics & participant counts.
  - Word Clouds for message content.
  - Timeseries graphs showing messages recorded over time.
  - Average message length per sender.
  - Emoji usage tracking!
- **Data Exporting**: Download detailed analytical reports and prediction logs (CSV format) directly from the browser.
- **Interactive UI**: Fully responsive Plotly charts with a robust Dark/Light mode theme toggle.

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Whatsapp-spam-detector.git
cd Whatsapp-spam-detector
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# On macOS/Linux use: source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model (If necessary)
A pre-trained model might already be in the `models/` directory. If you want to train it from scratch using your dataset:
```bash
python src/train_model.py
```
*Note: Ensure you have `spam.csv` (like the UCI SMS Spam Collection dataset) properly placed inside the `data/` folder before training.*

### 5. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 📱 How to Export Your WhatsApp Chat

To use the tool, you need to provide a `.txt` file of your chat.

**For Android:**
1. Open the WhatsApp chat you want to analyze.
2. Tap `⋮` (More options) → **More** → **Export Chat**.
3. Choose **"Without Media"**.
4. Save the generated `.txt` file and upload it to the application.

**For iOS (iPhone):**
1. Open the WhatsApp chat.
2. Tap the contact or group name at the top → **Export Chat**.
3. Choose **"Without Media"**.
4. Save the generated `.txt` file.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), NumPy
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (MultinomialNB, TfidfVectorizer)
- **Data Visualization:** [Plotly](https://plotly.com/python/), Matplotlib, WordCloud
- **NLP / Text Processing:** Regex, NLTK, Emoji

---

## 📂 Project Structure

```text
Whatsapp-spam-detector/
│
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── data/
│   ├── spam.csv               # Dataset used for training the ML Model
│   └── temp_chat.txt          # Temporary uploaded chat storage
│
├── models/
│   ├── spam_model.pkl         # Pickled MultinomialNB model
│   └── vectorizer.pkl         # Pickled TF-IDF vectorizer
│
└── src/
    ├── __init__.py
    ├── analysis.py            # Chat analytics (Wordcloud, emoji, timeline stats)
    ├── data_preprocessing.py  # Regex parsing of WhatsApp .txt files
    ├── Labelling.py           # Auto-labeling heuristics & dataset loader
    ├── predict.py             # Logic bridging the ML predictions and app
    └── train_model.py         # Script to ingest data and train the classifier
```

---

## 🧠 Under the Hood (The Model)

The prediction pipeline utilizes a two-step funnel:
1. **Rule-Based Engine:** Looks for highly indicative spam terminology combining URLs, typical bait words (like 'promo', 'reward', 'crypto', etc.).
2. **Probabilistic Engine:** A **Multinomial Naive Bayes** algorithm trained on English SMS spam databanks. Text is parsed into numerical arrays using **TF-IDF** (Term Frequency-Inverse Document Frequency) which identifies the importance of words contextualized against the background spam dataset.

