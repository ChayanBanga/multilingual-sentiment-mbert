# 🌍 Multilingual Sentiment Analyzer

Fine-tuned mBERT model for sentiment analysis across 100+ languages.
Trained only on English data — zero-shot transfer to other languages.

## 🚀 Live Demo
[Try it here](https://multilingual-sentiment-mbert-vszsjjh8jlgn7e3x7cgezz.streamlit.app/)

## 📊 Results
- Training Accuracy: 93.79%
- Test Accuracy: 87.68%
- Zero-shot Multilingual: 80% (Hindi, French, Spanish, German)

## 🛠️ Tech Stack
- Model: bert-base-multilingual-cased (mBERT)
- Training: Google Colab (T4 GPU)
- Dataset: Yelp Reviews (English, 5000 samples)
- Deployment: Streamlit Cloud
- News Integration: NewsAPI

## 🌐 How It Works
1. Fine-tuned mBERT on English sentiment data
2. mBERT's multilingual pre-training enables zero-shot transfer
3. Works on any language without additional training

## 🤗 Model
[Chayan123banga/multilingual-sentiment-mbert](https://huggingface.co/Chayan123banga/multilingual-sentiment-mbert)
