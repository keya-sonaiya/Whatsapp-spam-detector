# ================================
# WhatsApp Chat Analysis Module
# ================================

import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS 
from collections import Counter

# Import preprocessing functions
from src.data_preprocessing import load_chat, clean_chat


# -------------------------------
# 1. Basic Stats
# -------------------------------
def chat_stats(df):
    """Return total messages, unique participants, most active senders."""
    if df.empty:
        return 0, 0, pd.Series(dtype=int)
    total_msgs = len(df)
    participants = df['sender'].nunique()
    active_senders = df['sender'].value_counts()
    return total_msgs, participants, active_senders


# -------------------------------
# 2. Word Cloud
# -------------------------------
def generate_wordcloud(df, bg_color="#ffffff", max_words=200):
    if df.empty: return None
    text = " ".join(df['message'].astype(str).tolist())
    sw = set(STOPWORDS) | {"http","https","www","com"}
    wc = WordCloud(width=800, height=400, stopwords=sw,
                   background_color=bg_color, max_words=max_words,
                   collocations=False).generate(text)
    return wc


# -------------------------------
# 3. Messages Over Time
# -------------------------------
def messages_over_time(df, freq='D'):
    if df.empty: return pd.Series(dtype=int)
    tmp = df.copy()
    tmp['datetime'] = pd.to_datetime(tmp['datetime'], errors='coerce')
    tmp = tmp.dropna(subset=['datetime']).set_index('datetime')
    s = tmp.resample(freq).size()
    s.index = s.index.date  # optional: make index plain date objects
    return s



# -------------------------------
# 4. Average Message Length
# -------------------------------
def avg_message_length(df):
    """Return avg message length per sender."""
    if df.empty:
        return pd.Series(dtype=float)
    res = df.copy()
    res['msg_len'] = res['message'].astype(str).str.len()
    return res.groupby('sender')['msg_len'].mean().round(1).sort_values(ascending=False)



# -------------------------------
# 5. Top Words
# -------------------------------
def top_words(df, n=20):
    """Return top n common words."""
    if df.empty:
        return []
    text = " ".join(df['message'].tolist()).lower()
    tokens = re.findall(r"\b[^\d\W_]{3,}\b", text, flags=re.UNICODE)
    sw = set(STOPWORDS) | {"http","https","www","com"}
    tokens = [t for t in tokens if t not in sw]
    return Counter(tokens).most_common(n)




# -------------------------------
# 6. Emoji Usage
# -------------------------------


def emoji_usage(df, message_col="message", top_n=20):
    """
    Returns top N emojis used in the chat as a list of tuples: [(emoji, count), ...]
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "]+",
        flags=re.UNICODE
    )

    all_emojis = []

    # Loop through each message
    for msg in df[message_col].dropna():
        all_emojis.extend(emoji_pattern.findall(msg))

    if not all_emojis:
        return []

    emoji_counts = Counter(all_emojis)
    top_emojis = emoji_counts.most_common(top_n)

    return top_emojis



