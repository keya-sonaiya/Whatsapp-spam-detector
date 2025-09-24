# Whatsapp-spam-detector

A **Python & Streamlit** web app that detects spam messages in WhatsApp chats using **Machine Learning (Naive Bayes)** and **TF-IDF vectorization**. It classifies messages as **Spam** or **Ham** (non-spam) based on keywords and patterns learned from the dataset.  

---

## 🔹Features

- **SMS Spam Detection**: Classifies messages as Spam or Ham.  
- **Auto-labeling**: Flags messages with common spam keywords.  
- **Preprocessing**: Cleans text, removes URLs/numbers/punctuation, reduces repeated letters.  
- **TF-IDF Vectorization**: Converts text to numeric vectors for model input.  
- **Multinomial Naive Bayes**: Accurate message classification.  
- **Streamlit Web App**: Interactive interface for predictions.  
- **Evaluation Metrics**: Shows precision, recall, F1-score, and confusion matrix.  

---

## ⚡ Installation

1. **Clone the repository**

```bash
git clone https://github.com/keya-sonaiya/Whatsapp-spam-detector.git
cd Whatsapp-spam-detector
```
2. **Create and activate a virtual environment**
   
Windows
```
python -m venv venv
venv\Scripts\activate
```
macOS / Linux
```
python3 -m venv venv
source venv/bin/activate
```
3. **Install dependencies**
```
pip install -r requirements.txt
```
---

## 🚀Run the App
```
streamlit run app.py
```
---

## 📊 Dataset Details
- **Dataset source:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  



