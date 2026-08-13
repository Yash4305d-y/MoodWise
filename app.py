import os
import re
import string
import joblib
import nltk
import pandas as pd
import streamlit as st

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MoodWise | Emotion Classifier",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM CSS — MIDNIGHT EMBER THEME
# ============================================================

st.markdown(
    """
<style>
/* ---------- Global ---------- */
.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ---------- Hero ---------- */
.hero {
    text-align: center;
    padding: 1.5rem 0 1rem 0;
}

.hero-icon {
    font-size: 3.2rem;
    margin-bottom: 0.2rem;
    filter: grayscale(10%);
}

.hero-title {
    font-size: 3rem;
    font-weight: 850;
    letter-spacing: -1.8px;
    margin: 0;
    background: linear-gradient(90deg, #F59E0B, #F97316, #EAB308);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #A8A29E;
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
}

/* ---------- Section Titles ---------- */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin-top: 1.2rem;
    margin-bottom: 0.6rem;
    color: #E7E5E4;
}

/* ---------- Model Info ---------- */
.model-card {
    padding: 1rem 1.2rem;
    border-radius: 14px;
    border: 1px solid rgba(245, 158, 11, 0.16);
    background: rgba(245, 158, 11, 0.035);
    margin-bottom: 1rem;
}

.model-stat {
    text-align: center;
    padding: 0.4rem;
}

.model-stat-value {
    font-size: 1.25rem;
    font-weight: 750;
    color: #F5F5F4;
}

.model-stat-label {
    font-size: 0.75rem;
    color: #A8A29E;
}

/* ---------- Input ---------- */
div[data-testid="stTextArea"] textarea {
    border-radius: 14px !important;
    border: 1px solid rgba(168, 162, 158, 0.25) !important;
    padding: 1rem !important;
    font-size: 1rem !important;
    background: rgba(28, 25, 23, 0.45) !important;
    transition: all 0.2s ease;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #F59E0B !important;
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.14) !important;
}

/* ---------- Example Buttons ---------- */
div.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(168, 162, 158, 0.2);
    transition: all 0.2s ease;
    font-weight: 600;
    background: rgba(28, 25, 23, 0.35);
}

div.stButton > button:hover {
    transform: translateY(-2px);
    border-color: #F59E0B;
    color: #F59E0B;
}

/* ---------- Primary Button ---------- */
div.stButton > button[kind="primary"] {
    border-radius: 14px;
    font-size: 1.05rem;
    font-weight: 700;
    padding: 0.65rem 1rem;
    background: linear-gradient(135deg, #D97706, #EA580C);
    border: none;
    color: white;
    box-shadow: 0 6px 20px rgba(234, 88, 12, 0.18);
    transition: all 0.2s ease;
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(234, 88, 12, 0.28);
}

/* ---------- Metrics ---------- */
.metric-card {
    text-align: center;
    padding: 0.7rem;
    border-radius: 12px;
    background: rgba(245, 158, 11, 0.035);
    border: 1px solid rgba(245, 158, 11, 0.12);
}

.metric-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #F5F5F4;
}

.metric-label {
    font-size: 0.72rem;
    color: #A8A29E;
}

/* ---------- Result ---------- */
.result-card {
    padding: 1.5rem;
    border-radius: 18px;
    border: 1px solid rgba(245, 158, 11, 0.16);
    background: rgba(245, 158, 11, 0.035);
    margin-top: 1rem;
}

.result-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #A8A29E;
    font-weight: 700;
    margin-bottom: 0.4rem;
}

.emotion-name {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    color: #F5F5F4;
}

.confidence-text {
    font-size: 0.9rem;
    color: #A8A29E;
}

/* ---------- Emotion Badge ---------- */
.emotion-badge {
    display: inline-block;
    padding: 0.45rem 1rem;
    border-radius: 999px;
    color: white;
    font-size: 0.85rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ---------- Emotion Description ---------- */
.emotion-description {
    margin-top: 1rem;
    padding: 0.9rem 1rem;
    border-radius: 12px;
    background: rgba(245, 158, 11, 0.045);
    border-left: 3px solid #F59E0B;
    font-size: 0.9rem;
    line-height: 1.5;
    color: #D6D3D1;
}

/* ---------- Expander ---------- */
div[data-testid="stExpander"] {
    border-color: rgba(245, 158, 11, 0.14) !important;
    border-radius: 14px !important;
}

/* ---------- Divider ---------- */
hr {
    border-color: rgba(168, 162, 158, 0.15) !important;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: #78716C;
    font-size: 0.78rem;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(168, 162, 158, 0.15);
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================


@st.cache_resource
def load_artifacts():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

    model_dir = "model"

    model = joblib.load(os.path.join(model_dir, "emotion_model.pkl"))
    vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.pkl"))
    id_to_label = joblib.load(os.path.join(model_dir, "id_to_label.pkl"))

    return model, vectorizer, id_to_label


try:
    model, vectorizer, id_to_label = load_artifacts()
    stop_words = set(stopwords.words("english"))

    # ========================================================
    # TEXT CLEANING
    # ========================================================

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = "".join(char for char in text if not char.isdigit())
        text = text.encode("ascii", "ignore").decode("ascii")
        words = word_tokenize(text)
        words = [word for word in words if word.lower() not in stop_words]
        return " ".join(words)

    # ========================================================
    # EMOTION CONFIGURATION
    # ========================================================

    emotion_colors = {
        "sadness": "#64748B",
        "anger": "#EF4444",
        "love": "#F43F5E",
        "surprise": "#F59E0B",
        "fear": "#A78B5A",
        "joy": "#D97706",
    }

    emotion_icons = {
        "sadness": "😢",
        "anger": "😠",
        "love": "❤️",
        "surprise": "😲",
        "fear": "😨",
        "joy": "😊",
    }

    emotion_descriptions = {
        "sadness": "The text expresses feelings of sadness, loneliness, disappointment, or loss.",
        "anger": "The text expresses frustration, irritation, annoyance, or anger.",
        "love": "The text expresses affection, care, attraction, or strong emotional connection.",
        "surprise": "The text expresses astonishment, unexpectedness, or something remarkable.",
        "fear": "The text expresses worry, anxiety, nervousness, or fear.",
        "joy": "The text expresses happiness, excitement, satisfaction, or positive feelings.",
    }

    # ========================================================
    # HERO SECTION
    # ========================================================

    st.markdown(
        """<div class="hero">
<div class="hero-icon">🎭</div>
<div class="hero-title">MoodWise</div>
<div class="hero-subtitle">Understand the emotion hiding behind your words.</div>
</div>""",
        unsafe_allow_html=True,
    )

    # ========================================================
    # MODEL INFO
    # ========================================================

    with st.expander("🧠 About this model"):
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        info1, info2, info3 = st.columns(3)

        with info1:
            st.markdown(
                """<div class="model-stat">
<div class="model-stat-value">Logistic Regression</div>
<div class="model-stat-label">CLASSIFIER</div>
</div>""",
                unsafe_allow_html=True,
            )

        with info2:
            st.markdown(
                """<div class="model-stat">
<div class="model-stat-value">Bag of Words</div>
<div class="model-stat-label">FEATURE EXTRACTION</div>
</div>""",
                unsafe_allow_html=True,
            )

        with info3:
            st.markdown(
                """<div class="model-stat">
<div class="model-stat-value">~88.8%</div>
<div class="model-stat-label">TRAINING ACCURACY</div>
</div>""",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        st.caption(
            "Text is cleaned by removing URLs, punctuation, digits, "
            "and stopwords before being transformed into a Bag-of-Words "
            "representation."
        )

    # ========================================================
    # EXAMPLES
    # ========================================================

    st.markdown(
        '<div class="section-title">✨ Try a quick example</div>',
        unsafe_allow_html=True,
    )

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    examples = {
        "😊 Joy": "I am so happy and delighted with these amazing results!",
        "😠 Anger": "I am feeling so frustrated and annoyed with the delay!",
        "😢 Sadness": "I feel so lonely and hopeless today.",
        "😲 Surprise": "I was completely astonished by the unexpected news!",
    }

    cols = st.columns(4)

    def set_example_text(selected_text):
        st.session_state["input_text"] = selected_text

    for col, (label, text) in zip(cols, examples.items()):
        col.button(
            label,
            use_container_width=True,
            on_click=set_example_text,
            args=(text,),
        )

    # ========================================================
    # INPUT
    # ========================================================

    st.markdown(
        '<div class="section-title">💬 Your text</div>', unsafe_allow_html=True
    )

    user_text = st.text_area(
        "Enter sentence:",
        key="input_text",
        placeholder="e.g. I'm super excited about launching this new NLP application!",
        height=130,
        label_visibility="collapsed",
    )

    # ========================================================
    # LIVE METRICS
    # ========================================================

    if user_text.strip():
        words_count = len(user_text.split())
        char_count = len(user_text)
        read_time = max(1, round(words_count / 200))

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f"""<div class="metric-card">
<div class="metric-value">{words_count}</div>
<div class="metric-label">WORDS</div>
</div>""",
                unsafe_allow_html=True,
            )

        with m2:
            st.markdown(
                f"""<div class="metric-card">
<div class="metric-value">{char_count}</div>
<div class="metric-label">CHARACTERS</div>
</div>""",
                unsafe_allow_html=True,
            )

        with m3:
            st.markdown(
                f"""<div class="metric-card">
<div class="metric-value">{read_time}s</div>
<div class="metric-label">READ TIME</div>
</div>""",
                unsafe_allow_html=True,
            )

    # ========================================================
    # PREDICTION
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button(
        "Analyze Emotion →", type="primary", use_container_width=True
    )

    if predict_clicked:
        if not user_text.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            with st.spinner("Analyzing your text..."):
                cleaned = clean_text(user_text)
                vectorized = vectorizer.transform([cleaned])
                pred_id = model.predict(vectorized)[0]

                predicted_emotion = (
                    id_to_label.get(pred_id, str(pred_id))
                    if isinstance(id_to_label, dict)
                    else str(pred_id)
                )

                probabilities = model.predict_proba(vectorized)[0]

            # =================================================
            # RESULT
            # =================================================

            emotion_key = str(predicted_emotion).lower()
            bg_color = emotion_colors.get(emotion_key, "#78716C")
            icon = emotion_icons.get(emotion_key, "🎭")
            top_prob = max(probabilities) * 100

            st.markdown("---")

            result_col, chart_col = st.columns([0.85, 1.15], gap="large")

            # -------------------------------------------------
            # PRIMARY RESULT
            # -------------------------------------------------

            with result_col:
                st.markdown(
                    '<div class="result-label">Primary emotion</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""<div class="emotion-name">{icon} {str(predicted_emotion).title()}</div>
<div class="emotion-badge" style="background:{bg_color};">{top_prob:.1f}% confidence</div>""",
                    unsafe_allow_html=True,
                )

                description = emotion_descriptions.get(
                    emotion_key,
                    "The model detected this as the dominant emotion.",
                )

                st.markdown(
                    f"""<div class="emotion-description">{description}</div>""",
                    unsafe_allow_html=True,
                )

            # -------------------------------------------------
            # PROBABILITY CHART
            # -------------------------------------------------

            with chart_col:
                st.markdown(
                    '<div class="result-label">Emotion breakdown</div>',
                    unsafe_allow_html=True,
                )

                labels = [
                    id_to_label.get(cls, str(cls))
                    if isinstance(id_to_label, dict)
                    else str(cls)
                    for cls in model.classes_
                ]

                prob_df = pd.DataFrame(
                    {"Emotion": labels, "Probability": probabilities * 100}
                )

                prob_df = prob_df.sort_values("Probability", ascending=False)
                chart_df = prob_df.set_index("Emotion")

                st.bar_chart(chart_df, height=230)

            # =================================================
            # DETAILED SCORES
            # =================================================

            with st.expander("📊 View detailed confidence scores"):
                display_df = prob_df.copy()
                display_df["Probability"] = display_df["Probability"].map(
                    lambda x: f"{x:.2f}%"
                )
                display_df = display_df.rename(
                    columns={"Probability": "Confidence"}
                )

                st.dataframe(
                    display_df, use_container_width=True, hide_index=True
                )

    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        """<div class="footer">
Built with Python · Scikit-learn · NLTK · Streamlit
<br><br>
🎭 MoodWise · NLP Text Classification
</div>""",
        unsafe_allow_html=True,
    )

# ============================================================
# ERROR HANDLING
# ============================================================

except FileNotFoundError as e:
    st.error(f"Missing model artifact: {e}")
    st.info(
        "Make sure emotion_model.pkl, vectorizer.pkl, "
        "and id_to_label.pkl are inside the model/ folder."
    )