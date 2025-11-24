import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. Load and Prepare Dataset ---
print("="*30)
print("--- Loading dataset from fake_or_real_news.csv... ---")
try:
    df = pd.read_csv('fake_or_real_news.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'fake_or_real_news.csv' not found. Please ensure it is in the same directory.")
    exit()

# Extract features (text) and labels
texts = df['text'].values
labels = df['label'].values

# --- 2. Split Data into Training and Testing Sets ---
print("--- Splitting data into 80% training and 20% testing sets... ---")
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# --- 3. Feature Extraction and Model Training ---
print("--- Applying TF-IDF Vectorization and training Logistic Regression model... ---")

# Initialize the vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data
X_train_vec = vectorizer.fit_transform(X_train_text)

# Only transform the test data
X_test_vec = vectorizer.transform(X_test_text)

# Initialize and train the model
model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
model.fit(X_train_vec, y_train)
print("--- Model training complete. ---")

# --- 4. Model Evaluation ---
print("--- Evaluating model performance on the test set... ---")
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print("="*30)

# --- 5. Save the Trained Model and Vectorizer ---
print("--- Saving vectorizer to 'tfidf_vectorizer.joblib'... ---")
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("--- Saving model to 'logistic_regression_model.joblib'... ---")
joblib.dump(model, 'logistic_regression_model.joblib')
print("--- Artifacts saved successfully. ---")
print("="*30)

# --- 6. FastAPI Application Setup ---
app = FastAPI(
    title="Fake News Detector API",
    description="An API to detect whether a news article is REAL or FAKE using a trained Logistic Regression model.",
    version="1.0.0"
)

class Article(BaseModel):
    text: str

@app.post("/detect/", summary="Detect Fake News")
def detect_fake_news(article: Article):
    """
    Analyzes a news article's text to predict if it's REAL or FAKE.
    - **text**: The full text of the news article.
    \f
    :param article: An Article object containing the text to be analyzed.
    :return: A dictionary with the original text and the AI analysis result.
    """
    # Transform the input text using the loaded vectorizer
    vectorized_text = vectorizer.transform([article.text])
    
    # Predict the label
    prediction_label = model.predict(vectorized_text)[0]
    
    # Get the prediction probabilities
    prediction_confidence = model.predict_proba(vectorized_text)[0]
    
    # Extract the confidence for the predicted class
    confidence = np.max(prediction_confidence)

    ai_result = {
        "prediction": prediction_label,
        "confidence": confidence
    }

    return {"original_text": article.text, "ai_analysis": ai_result}

@app.get("/", summary="Root Endpoint")
def read_root():
    """
    A simple root endpoint to confirm the API is running.
    """
    return {"message": "Fake News Detector API is running. Navigate to /docs for the interactive API documentation."}