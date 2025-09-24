import pandas as pd

def auto_label(df):
    """
    Adds 'auto_spam' column = True if message looks like spam
    """
    spam_keywords = (
    r"http|https|www|\.com|\.net|\.org|\.in|free|offer|win|money|prize|lottery|click|"
    r"winner|gift|trial|bonus|voucher|urgent|subscribe|deal|congratulations|won|"
    r"discount|limited|cash|reward|claim|exclusive|promo|promotion|guarantee|risk-free|"
    r"cheap|save|bargain|sale|earn|bitcoin|crypto|investment|loan|credit|referral|"
    r"pay|payment|account|password|verify|alert|notification|coupon|"
    r"subscribe now|join now|act fast|limited time|buy now|click here|get it now|"
    r"visit|register|sign up|apply|cash prize|instant cash|free trial|special offer|"
    r"congratulations you won|winner announcement"
    )
    df['auto_spam'] = df['message'].str.contains(spam_keywords, case=False, na=False)
    df['auto_spam_label'] = df['auto_spam'].map({True: "Spam", False: "Ham"})
    return df


def load_external_dataset(path="D:/ML/whatsapp-spam/data/spam.csv"):
    """
    Loads UCI SMS Spam dataset (spam.csv).
    Assumes first two columns are: [label, message].
    Ignores extra columns.
    Maps ham/spam → 0/1.
    """
    ext_df = pd.read_csv(path, encoding="latin-1", sep='\t', header=None, names=['label','message'], on_bad_lines='skip')
    ext_df.dropna(inplace=True)
    ext_df['label'] = ext_df['label'].str.strip().map({'ham': 0, 'spam': 1})
    return ext_df
