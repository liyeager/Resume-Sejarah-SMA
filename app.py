import streamlit as st
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi stopword dan stemmer
factory_stop = StopWordRemoverFactory()
stopwords = factory_stop.get_stop_words()
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()

st.set_page_config(page_title="LSA Summarizer", layout="wide")

# Custom CSS: tema pastel + input box lebih ramping
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #fce4ec, #e3f2fd);
        }
        .block-container {
            padding: 2rem;
            border-radius: 15px;
            background-color: #ffffffcc;
            color: #333;
        }
        .stTextArea textarea {
            background-color: #ffffff;
            color: black;
            width: 60% !important;
            margin: auto;
        }
        .stButton>button {
            color: white;
            background-color: #ec407a;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("LSA Summarizer")

input_text = st.text_area("Masukkan teks panjang untuk diringkas:", height=100)

if st.button("Ringkas Teks"):
    if input_text:
        # Pisahkan kalimat
        sentences = re.split(r'(?<=[.!?])\s+', input_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        total_kalimat = len(sentences)

        # Preprocessing
        preprocessed_sentences = []
        for kalimat in sentences:
            lower = kalimat.lower()
            tokens = re.findall(r'\b\w+\b', lower)
            removed = [w for w in tokens if w not in stopwords]
            stemmed = [stemmer.stem(w) for w in removed]
            joined = " ".join(stemmed)
            preprocessed_sentences.append(joined)

        # TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(preprocessed_sentences)

        # SVD
        svd = TruncatedSVD(n_components=1, random_state=42)
        X_svd = svd.fit_transform(X)
        scores = X_svd[:, 0]

        # Ringkasan 10%
        n = max(1, int(total_kalimat * 0.1))
        top_indices = np.argsort(-scores)[:n]
        summary = " ".join([sentences[i] for i in sorted(top_indices)])

        # Tampilkan jumlah kalimat dan ringkasan
        st.markdown(f"**Jumlah kalimat sebelum diringkas:** {total_kalimat}")
        st.markdown(f"**Jumlah kalimat setelah diringkas (10%):** {n}")

        st.subheader("Ringkasan Teks")
        st.markdown(
            f"<div style='background-color:#fce4ec; color:#333; padding: 1rem; border-radius: 10px; font-size: 16px;'>{summary}</div>",
            unsafe_allow_html=True)
    else:
        st.warning("Teks tidak boleh kosong.")
