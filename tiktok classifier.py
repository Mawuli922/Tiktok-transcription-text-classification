import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import joblib  # For saving the model

# Load the dataset
tiktok_df = pd.read_csv("tiktok_dataset.csv")

# Data Exploration and Cleaning
print(tiktok_df.info())
print(tiktok_df["claim_status"].value_counts(dropna=False))

# Handle missing values
tiktok_df = tiktok_df.dropna(subset=["video_transcription_text"])  # Drop only rows with missing text

# Convert claim status to numeric
tiktok_df["claim_status_numeric"] = tiktok_df["claim_status"].apply(lambda x: 1 if x == "claim" else 0)

# Feature Engineering
tiktok_df["x_var"] = tiktok_df["video_transcription_text"] + " " + tiktok_df["verified_status"] + " " + tiktok_df["author_ban_status"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95, min_df=5, ngram_range=(1,2))  # Improved vectorizer with tuning
X_transformed = vectorizer.fit_transform(tiktok_df["x_var"])

y = tiktok_df["claim_status_numeric"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, stratify=y, random_state=42)

# Model Training and Tuning
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200)
}

# Hyperparameter Tuning Example
param_grid = {
    'Logistic Regression': {'C': [0.01, 0.1, 1, 10]},
    'Naive Bayes': {'alpha': [0.01, 0.1, 1]},
    'Random Forest': {'n_estimators': [100, 200], 'min_samples_leaf': [0.05, 0.1]}
}

best_models = {}
for model_name in models:
    grid = GridSearchCV(models[model_name], param_grid[model_name], cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_models[model_name] = grid.best_estimator_
    print(f"Best params for {model_name}: {grid.best_params_}")

# Model Evaluation
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title(f"Confusion Matrix for {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Save the best model
best_model = max(best_models, key=lambda x: accuracy_score(y_test, best_models[x].predict(X_test)))
joblib.dump(best_models[best_model], f"{best_model}_model.pkl")
print(f"Saved the best model: {best_model}")

joblib.dump(vectorizer, "vectorizer.pkl")