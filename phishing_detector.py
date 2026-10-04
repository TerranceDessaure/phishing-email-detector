import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Load the data
df = pd.read_csv("data/Phishing_Email.csv")
df = df.drop(columns=["Unnamed: 0"])
df = df.rename(columns={"Email Text": "text", "Email Type": "label"})

# Clean the data
df = df.dropna()
df["label"] = (df["label"] == "Phishing Email").astype(int)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)  # replace links
    text = re.sub(r"\S+@\S+", " emailtoken ", text)              # replace email addresses
    text = re.sub(r"\d+", " numtoken ", text)                    # replace numbers
    text = re.sub(r"[^a-z\s]", " ", text)                        # remove punctuation
    return re.sub(r"\s+", " ", text).strip()                     # collapse extra spaces

df["text"] = df["text"].apply(clean_text)
df = df[df["text"] != ""]                                # remove emails that are now empty
df = df.drop_duplicates(subset="text")                   # remove duplicate emails

print(f"Emails after cleaning: {len(df):,}")
print(df["label"].value_counts().rename({0: "Safe", 1: "Phishing"}))
print("\nExample cleaned email:\n", df["text"].iloc[0][:300])