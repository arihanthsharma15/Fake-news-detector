# Fake News Detector using Machine Learning

This project is a web application that classifies news articles as either "REAL" or "FAKE". It uses a Machine Learning model trained on a publicly available dataset and provides a real-time detection interface.

## Features
- **Machine Learning Model:** Utilizes a **Logistic Regression** classifier for prediction.
- **NLP Technique:** Employs **TF-IDF** for text feature extraction.
- **Client-Server Architecture:**
    - **Backend:** A robust REST API built with **FastAPI**.
    - **Frontend:** An interactive and user-friendly web interface built with **Streamlit**.
- **Real-time Analysis:** Allows users to paste news text and get an instant prediction.

---

## Technology Stack
- **Language:** Python 3.9+
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **ML/NLP Libraries:** Scikit-learn, Pandas, Joblib, NLTK
- **Data:** [Kaggle Fake or Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Git

### Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/arihanthsharma15/Fake-news-detector.git
    cd Fake-news-detector
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset:**
    Download the `fake_or_real_news.csv` file from the [Kaggle Dataset Link](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place it in the root directory of the project.

5.  **Train the Model and Generate Artifacts:**
    Run the `main.py` script once to train the model and save the `*.joblib` files.
    ```bash
    python main.py
    ```
    You can stop this script with `Ctrl+C` after the model files are saved.

---

## How to Run

The application runs in two parts. You will need two separate terminals.

1.  **Run the Backend Server (Terminal 1):**
    ```bash
    uvicorn main:app --reload
    ```
    The API will be running at `http://127.0.0.1:8000`.

2.  **Run the Frontend Application (Terminal 2):**
    ```bash
    streamlit run ui.py
    ```
    Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

---

## Project Structure
```
.
├── main.py           # FastAPI backend: trains model, saves artifacts, and serves API
├── ui.py             # Streamlit frontend: user interface for prediction
├── requirements.txt  # Project dependencies
├── .gitignore        # Files and folders to be ignored by Git
└── README.md         # This file
```