import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests

# Page config
st.set_page_config(
    page_title="Multilingual Sentiment Analyzer",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Multilingual Sentiment Analyzer")
st.markdown("Powered by fine-tuned mBERT — works in 100+ languages")

# Load model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Chayan123banga/multilingual-sentiment-mbert")
    model = AutoModelForSequenceClassification.from_pretrained("Chayan123banga/multilingual-sentiment-mbert")
    model.eval()
    return tokenizer, model

with st.spinner("Loading model..."):
    tokenizer, model = load_model()

# Prediction function
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item() * 100
    return pred, confidence

# ── Section 1: Custom Text ──
st.header("📝 Analyze Your Text")
user_input = st.text_area("Enter text in any language:", height=100)

if st.button("Analyze", type="primary"):
    if user_input.strip():
        pred, confidence = predict(user_input)
        if pred == 1:
            st.success(f"✅ POSITIVE — {confidence:.1f}% confident")
        else:
            st.error(f"❌ NEGATIVE — {confidence:.1f}% confident")
    else:
        st.warning("Please enter some text!")

st.divider()

# ── Section 2: Live News ──
st.header("📰 Live News Sentiment")
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")

topic = st.text_input("Enter news topic:", value="artificial intelligence")

if st.button("Fetch & Analyze News"):
    if not NEWS_API_KEY:
        st.error("NewsAPI key not found! Add it to Streamlit secrets.")
    else:
        with st.spinner("Fetching news..."):
            url = f"https://newsapi.org/v2/everything?q={topic}&pageSize=10&apiKey={NEWS_API_KEY}"
            response = requests.get(url)
            data = response.json()

        if data.get("articles"):
            positive = 0
            negative = 0

            for article in data["articles"]:
                title = article.get("title", "")
                if not title:
                    continue
                pred, confidence = predict(title)

                if pred == 1:
                    positive += 1
                    bg, border, icon = "#0f3d24", "#4ade80", "✅"
                else:
                    negative += 1
                    bg, border, icon = "#4c0519", "#f87171", "❌"

                st.markdown(
                    f"""
                    <div style="
                        background:{bg};
                        border:1px solid {border};
                        border-radius:8px;
                        padding:0.8rem 1rem;
                        margin-bottom:0.6rem;
                        white-space:normal;
                        word-wrap:break-word;
                        overflow-wrap:break-word;
                        line-height:1.5;
                        color:#e2e8f0;
                    ">
                        {icon} {title} &nbsp;<b>({confidence:.1f}%)</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.divider()
            total = positive + negative
            st.metric("Overall Mood", 
                      "🟢 Positive" if positive > negative else "🔴 Negative",
                      f"{positive} positive, {negative} negative out of {total} articles")
        else:
            st.error("No articles found. Try a different topic.")

st.divider()
st.caption("Trained on English Yelp reviews | Zero-shot multilingual via mBERT")
