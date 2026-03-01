import streamlit as st
import streamlit.components.v1 as components
import joblib
import os
import math

# Set page config
st.set_page_config(page_title="TruthLens AI - Fake News Detector", page_icon="🔍", layout="centered")

# Read the original style.css
css_path = os.path.join(os.path.dirname(__file__), 'static', 'style.css')
with open(css_path, 'r', encoding='utf-8') as f:
    original_css = f.read()

# Inject full CSS: original style.css + Streamlit overrides
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&display=swap');

    {original_css}

    /* === Streamlit Overrides === */
    .stApp {{
        background-color: #0f172a !important;
        font-family: 'Outfit', sans-serif !important;
    }}
    header[data-testid="stHeader"] {{ display: none !important; }}
    #MainMenu {{ display: none !important; }}
    .block-container {{
        padding-top: 1rem !important;
        max-width: 800px !important;
    }}

    /* Text Area */
    .stTextArea textarea {{
        width: 100% !important;
        min-height: 200px !important;
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        color: #f8fafc !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }}
    .stTextArea textarea:focus {{
        border-color: rgba(59, 130, 246, 0.5) !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1) !important;
    }}
    .stTextArea textarea::placeholder {{
        color: rgba(148, 163, 184, 0.5) !important;
    }}
    .stTextArea label, .stTextArea div[data-testid="stWidgetLabel"] {{
        display: none !important;
    }}

    /* Button */
    div.stButton > button {{
        background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 20px -10px rgba(59, 130, 246, 0.5) !important;
        float: right !important;
    }}
    div.stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 25px -10px rgba(59, 130, 246, 0.5) !important;
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    }}
    div.stButton {{
        display: flex !important;
        justify-content: flex-end !important;
    }}

    /* Warning/Error */
    .stAlert {{ border-radius: 12px !important; }}
</style>
""", unsafe_allow_html=True)

# Animated background + Header (rendered as raw HTML component)
components.html(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&display=swap');
    {original_css}
    body {{
        background: transparent !important;
        display: flex;
        justify-content: center;
        min-height: auto;
    }}
    .app-container {{
        padding: 1rem 0 0 0;
    }}
</style>
<div class="animated-bg"></div>
<div class="app-container">
    <header class="app-header">
        <h1 class="logo">Truth<span class="gradient-text">Lens</span></h1>
        <p class="subtitle">Next-Generation AI Authentication Engine</p>
    </header>
</div>
""", height=150)


# Load model
@st.cache_resource
def load_models():
    model_path = 'fake_news_model.joblib'
    vectorizer_path = 'tfidf_vectorizer.joblib'
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        return joblib.load(model_path), joblib.load(vectorizer_path)
    return None, None

clf, vectorizer = load_models()

if not clf or not vectorizer:
    st.error("Model or vectorizer not found. Please train the model first.")
else:
    text_input = st.text_area("", height=200,
                               placeholder="Paste article text here to verify its authenticity...",
                               label_visibility="collapsed")

    submit_button = st.button("⚡ Analyze Content")

    if submit_button:
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing content authenticity..."):
                try:
                    text_vectorized = vectorizer.transform([text_input])
                    prediction = clf.predict(text_vectorized)[0]

                    dist = abs(clf.decision_function(text_vectorized)[0])
                    confidence = 1 / (1 + math.exp(-dist))
                    confidence_percent = round(confidence * 100, 1)
                    if confidence_percent < 50:
                        confidence_percent = 50 + (confidence_percent / 2)
                    if confidence_percent > 99.9:
                        confidence_percent = 99.9

                    is_fake = prediction == 1

                    if is_fake:
                        badge_class = "fake"
                        badge_text = "Fake News Detected"
                        bar_color = "linear-gradient(90deg, #ef4444, #b91c1c)"
                        insight = f"The AI model detected linguistic patterns, tone variations, and structural anomalies highly correlated with fabricated content. Confidence level: {confidence_percent}%."
                    else:
                        badge_class = "real"
                        badge_text = "Verified Authentic"
                        bar_color = "linear-gradient(90deg, #10b981, #047857)"
                        insight = f"The analysis indicates this article utilizes credible journalistic structures, consistent tone, and factual framing. Confidence level: {confidence_percent}%."

                    # Render results using components.html for proper HTML rendering
                    results_html = f"""
                    <style>
                        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&display=swap');
                        {original_css}
                        body {{
                            background: transparent !important;
                            display: block;
                            min-height: auto;
                        }}
                        .app-container {{
                            padding: 0;
                        }}
                    </style>
                    <div class="glass-card" style="animation: slide-up 0.5s cubic-bezier(0.16, 1, 0.3, 1);">
                        <div class="result-header">
                            <h2 class="result-title">Analysis Complete</h2>
                            <div class="badge {badge_class}">{badge_text}</div>
                        </div>
                        <div class="metrics">
                            <div class="metric-label">
                                <span>Confidence Score</span>
                                <span>{confidence_percent}%</span>
                            </div>
                            <div class="progress-track">
                                <div class="progress-fill" style="width: {confidence_percent}%; background: {bar_color};"></div>
                            </div>
                            <p class="insight-text">{insight}</p>
                        </div>
                    </div>
                    """
                    components.html(results_html, height=280)

                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")

# Footer
components.html(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&display=swap');
    body {{ background: transparent !important; font-family: 'Outfit', sans-serif; }}
    footer {{
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        opacity: 0.7;
        padding-top: 1rem;
    }}
</style>
<footer>
    <p>Powered by LinearSVC &amp; TF-IDF • IBM Project 2026</p>
</footer>
""", height=60)
