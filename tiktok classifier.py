import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

tiktok_df = pd.read_csv("tiktok_dataset.csv")

print(tiktok_df.info())

print(tiktok_df["claim_status"].value_counts(dropna=False))

print(tiktok_df.isna().sum().sort_values())
print(len(tiktok_df))

tiktok_df = tiktok_df.dropna(axis=0)

print(tiktok_df.isna().sum().sort_values())
print(len(tiktok_df))
# Convert claim status to numeric value
tiktok_df["claim_status_numeric"] = tiktok_df["claim_status"].apply(lambda x: 1 if x == "claim" else 0)

print(tiktok_df["claim_status_numeric"].value_counts(dropna=False))

# Feature Engineering

tiktok_df["x_var"] = tiktok_df["video_transcription_text"] + " " + tiktok_df["verified_status"] + " " + tiktok_df["author_ban_status"]

vectorizer = TfidfVectorizer(stop_words="english")

X = tiktok_df["x_var"]

X_transformed = vectorizer.fit_transform(X)

print(f"Shape of independent variable before transformation: {X.shape}")

print(f"Shape of independent variables after transformation: {X_transformed.shape}")

y = tiktok_df["claim_status_numeric"]

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, stratify=y, random_state=42)


lr = LogisticRegression(C=0.1)
nb = MultinomialNB()
rf = RandomForestClassifier(n_estimators=200, criterion="gini", min_samples_leaf=0.10)

model_dict = {"lr": lr, "nb": nb, "rf": rf}

accuracy_dict = {}

for key, value in model_dict.items():
    model = value.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_dict[key] = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap="Greens")
    plt.show()

print(accuracy_dict)

