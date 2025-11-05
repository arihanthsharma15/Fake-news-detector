#  AI Fake News Detector

A fast and stable fake news detector built with FastAPI, Streamlit, and a custom-trained Scikit-learn model. This project runs 100% locally, with no need for external APIs or internet for analysis.

---

###  **Key Features**

-   **Custom AI Model:** Trained from scratch on a real news dataset.
-   **Interactive UI:** Simple and clean interface built with Streamlit.
-   **Fast & Local:** Get instant predictions without any external dependency.
-   **Confidence Score:** See how confident the model is in its verdict.

---

###  **Tech Stack**

-   **Backend:** FastAPI
-   **Frontend:** Streamlit
-   **Machine Learning:** Scikit-learn, Pandas

---

###  **How to Run**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/arihanthsharma15/Fake-news-detector.git
    cd Fake-news-detector
    ```

2.  **Create a Conda environment and install dependencies:**
    ```bash
    conda create -n final_env python=3.9 -y
    conda activate final_env
    pip install fastapi "uvicorn[standard]" streamlit scikit-learn pandas
    ```

3.  **Run the Backend (Terminal 1):**
    ```bash
    uvicorn main:app --reload
    ```

4.  **Run the Frontend (Terminal 2):**
    ```bash
    streamlit run ui.py
    ```

---
*Project by Arihanth Sharma.*
