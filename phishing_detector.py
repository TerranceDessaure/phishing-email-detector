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

# Split into training and test sets

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2,          # 20% held back for testing
    random_state=42,        # makes the split the same every run
    stratify=df["label"],   # keeps the 62/38 ratio in both sets
)

# Build and train the model
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                              min_df=2, max_features=50000)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)

# Evaluate on the test set
predictions = model.predict(X_test)
print("\n", classification_report(y_test, predictions, target_names=["Safe", "Phishing"]))

ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, display_labels=["Safe", "Phishing"], cmap="Blues")
plt.title("Phishing Detector: Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved confusion_matrix.png")

# Security analysis: top phishing indicators ----------
words = model.named_steps["tfidf"].get_feature_names_out()
weights = model.named_steps["clf"].coef_[0]
ranked = pd.Series(weights, index=words).sort_values()

print("\nTop 15 phishing indicators:")
print(ranked.tail(15)[::-1].round(2).to_string())
print("\nTop 15 safe indicators:")
print(ranked.head(15).round(2).to_string())

plt.figure(figsize=(8, 6))
ranked.tail(15).plot(kind="barh", color="#c0392b")
plt.title("Top 15 Words and Phrases Linked to Phishing")
plt.xlabel("Model weight (higher = more suspicious)")
plt.savefig("top_phishing_words.png", dpi=150, bbox_inches="tight")
print("\nSaved top_phishing_words.png")