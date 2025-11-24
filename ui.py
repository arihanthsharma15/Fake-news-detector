import streamlit as st
import requests

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Truth-Lens AI",
    page_icon="🤖",
    layout="wide", # Use the full screen width
    initial_sidebar_state="expanded" # Keep the sidebar open initially
)

# --- Backend URL ---
BACKEND_URL = "http://127.0.0.1:8000/detect/"

# --- Sidebar Content ---
with st.sidebar:
    st.title("🤖 Truth-Lens AI")
    st.info("This is a Fake News Detector built with a custom-trained NLP model. It runs 100% locally.")
    st.markdown("---")
    st.write("### How it works:")
    st.write("1. **Backend:** FastAPI")
    st.write("2. **Frontend:** Streamlit")
    st.write("3. **AI Model:** Scikit-learn (Logistic Regression)")
    st.markdown("---")

# --- Main Page Content ---
st.title("📰 Fake News Detection Engine")
st.write("Enter a news headline or the full text of an article below to check its authenticity.")

# User Input
user_input = st.text_area("Enter News Text Here:", "A man from Florida claims he was abducted by aliens.", height=250)

# "Detect" button and logic
if st.button("Analyze Authenticity", type="primary", use_container_width=True):
    if user_input.strip():
        # Create two columns for the spinner and the result
        col1, col2 = st.columns([1, 2])
        
        with col1:
            with st.spinner("AI is analyzing..."):
                try:
                    payload = {"text": user_input}
                    response = requests.post(BACKEND_URL, json=payload)
                except requests.exceptions.RequestException:
                    st.error("Connection Error! Is the backend running?")
                    response = None # Set response to None if connection fails

        if response and response.status_code == 200:
            result = response.json()
            ai_analysis = result.get("ai_analysis")

            with col2:
                if "prediction" in ai_analysis:
                    st.balloons()
                    label = ai_analysis.get('prediction').upper()
                    score = ai_analysis.get('confidence', 0)

                    if "REAL" in label:
                        st.success(f"**Verdict: REAL NEWS** (Confidence: {score:.1%})")
                        st.progress(score)
                    else: # FAKE
                        st.error(f"**Verdict: FAKE NEWS** (Confidence: {score:.1%})")
                        st.progress(score)
                    
                    with st.expander("Show More Details"):
                        st.write("The model analyzed the text and predicted the following:")
                        st.json(ai_analysis)
                else:
                    st.error("Backend returned an unexpected format.")
                    st.json(ai_analysis)

    else:
        st.warning("Please enter a headline to detect.")