import streamlit as st
import joblib
import os
import math

# Set page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0px;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-real {
        color: #2E7D32;
        font-weight: bold;
        font-size: 1.5rem;
        padding: 15px;
        background-color: #E8F5E9;
        border-radius: 8px;
        border-left: 5px solid #2E7D32;
        text-align: center;
        margin-top: 10px;
    }
    .result-fake {
        color: #C62828;
        font-weight: bold;
        font-size: 1.5rem;
        padding: 15px;
        background-color: #FFEBEE;
        border-radius: 8px;
        border-left: 5px solid #C62828;
        text-align: center;
        margin-top: 10px;
    }
    .stTextArea textarea {
        font-size: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Fake News Detector 🕵️</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Paste an article below to check if it\'s real or fake.</p>', unsafe_allow_html=True)

# Load the model and vectorizer at startup
@st.cache_resource
def load_models():
    model_path = 'fake_news_model.joblib'
    vectorizer_path = 'tfidf_vectorizer.joblib'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        clf = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return clf, vectorizer
    else:
        return None, None

clf, vectorizer = load_models()

if not clf or not vectorizer:
    st.error("Error: Model (`fake_news_model.joblib`) or vectorizer (`tfidf_vectorizer.joblib`) not found. Please train the model first.")
else:
    text_input = st.text_area("Article Text:", height=250, placeholder="Paste the news article text here...")
    
    if st.button("Analyze Article", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    # Vectorize the input text
                    text_vectorized = vectorizer.transform([text_input])
                    prediction = clf.predict(text_vectorized)[0]
                    
                    # Calculate a pseudo-confidence score using decision_function
                    dist = abs(clf.decision_function(text_vectorized)[0])
                    # Simple sigmoid-like normalization for confidence visualization
                    confidence = 1 / (1 + math.exp(-dist))
                    confidence_percent = round(confidence * 100, 2)
                    
                    st.markdown("---")
                    st.subheader("Analysis Result")
                    
                    # Columns for better layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 0:
                            st.markdown('<p class="result-real">✅ REAL NEWS</p>', unsafe_allow_html=True)
                        else:
                            st.markdown('<p class="result-fake">❌ FAKE NEWS</p>', unsafe_allow_html=True)
                            
                    with col2:
                        st.metric(label="Confidence Score", value=f"{confidence_percent}%")
                        st.progress(confidence)
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
