import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# --- 1. Load the REAL Dataset ---
print("="*30)
print("--- Loading real dataset from text_db.csv... ---")
try:
    df = pd.read_csv('fake_or_real_news.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("!!! ERROR: text_db.csv not found! Make sure it's in the same folder. !!!")
    exit()

# The dataset has 'text' and 'label' columns. We will use them.
# We will use only a part of the dataset to keep training fast.
# You can increase this number to make the model smarter.
subset_size = 2000 
df_subset = df.head(subset_size)

texts = df_subset['text'].values
labels = df_subset['label'].values

# --- 2. The Tools (No change here) ---
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
model = LogisticRegression()

# --- 3. Training the Model on REAL Data ---
print(f"--- Training our model on {len(texts)} real news articles... ---")
X_train = vectorizer.fit_transform(texts)
model.fit(X_train, labels)
print("--- Model training complete! ✅ ---")
print("="*30)

# --- FastAPI App Setup (No change here) ---
app = FastAPI(title="Real Fake News Detector")

class Article(BaseModel):
    text: str

# --- API Endpoint (No change here) ---
@app.post("/detect/")
def detect_fake_news(article: Article):
    X_new = vectorizer.transform([article.text])
    prediction_code = model.predict(X_new)[0]
    confidence = np.max(model.predict_proba(X_new)[0])
    prediction_label = "REAL" if prediction_code == 1 else "FAKE"

    ai_result = {
        "prediction": prediction_label,
        "confidence": confidence
    }

    return {"original_text": article.text, "ai_analysis": ai_result}

@app.get("/")
def read_root():
    return {"message": "My own Fake News Detector (trained on real data) is running!"}