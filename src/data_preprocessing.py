# ================================
# WhatsApp Chat Preprocessing Module
# ================================

import re
import pandas as pd
from datetime import datetime

def load_chat(file_path):
    """
    Loads WhatsApp chat from exported .txt file.
    Handles multiline messages and extracts datetime, sender, and message.
    Returns a DataFrame with columns: [datetime, sender, message]
    """
    chat_data = []

    # Regex pattern for WhatsApp messages:
    # Handles dates, times (with optional AM/PM), dash or EN dash, sender, message
    line_pattern = re.compile(
        r"^(\d{1,2}/\d{1,2}/\d{2,4}), "
        r"(\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?)\s?[–-] (.*?): (.*)$"
    )

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_sender = None
    current_message = []
    current_datetime = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = line_pattern.match(line)
        if match:
            # Save previous message
            if current_sender and current_message:
                chat_data.append([current_datetime, current_sender, " ".join(current_message)])

            date_str, time_str, sender, message = match.groups()

            # Parse datetime safely with multiple formats
            dt = None
            for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y %I:%M %p",
                        "%d/%m/%y %H:%M", "%d/%m/%y %I:%M %p"):
                try:
                    dt = datetime.strptime(f"{date_str} {time_str}", fmt)
                    break
                except:
                    continue
            current_datetime = dt
            current_sender = sender
            current_message = [message]
        else:
            # Continuation of previous message
            if current_message is not None:
                current_message.append(line)

    # Save last message
    if current_sender and current_message:
        chat_data.append([current_datetime, current_sender, " ".join(current_message)])

    # Convert to DataFrame
    df = pd.DataFrame(chat_data, columns=["datetime", "sender", "message"])

    # Drop system messages (like encryption notices)
    df = df[~df["message"].str.contains("end-to-end encryption", case=False, na=False)]

    # Remove empty rows
    df = df[df["message"].str.strip() != ""].reset_index(drop=True)

    if df.empty:
        raise ValueError("No messages loaded. Check your WhatsApp chat format.")

    return df


def clean_chat(df: pd.DataFrame) -> pd.DataFrame:
    """Clean chat DataFrame: remove empty messages and strip text."""
    if not df.empty:
        df.dropna(subset=["message"], inplace=True)
        df["message"] = df["message"].str.strip()
    return df
