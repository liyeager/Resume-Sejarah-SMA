import streamlit as st
import string
import numpy as np
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Hapus semua nltk.download dan word_tokenize

# Inisialisasi tools lokal
stemmer = StemmerFactory().create_stemmer()
stop_words = set(StopWordRemoverFactory().get_stop_words())
tokenizer = RegexpTokenizer(r'\w+')

# Streamlit Setup
st.set_page_config(page_title="LSA Text Summarizer", layout="wide")

st.markdown("""
    <style>
        html, body {
            font-family: 'Comic Sans MS', cursive, sans-serif;
            background-color: #fff0f5;
        }
        h1, h2, h3 {
            color: #d63384;
        }
        .stButton>button {
            background-color: #ffe6f0;
            color: #d63384;
            border: 2px solid #d63384;
            border-radius: 12px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💗 LSA Text Summarizer - Input Teks Manual")

# Input dari pengguna
input_text = st.text_area("📝 Masukkan teks untuk diringkas:")

if st.button("🔍 Ringkas Sekarang") and input_text.strip():
    # 1. Split jadi kalimat manual
    sentences = re.split(r'(?<=[.!?])\s+', input_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    st.subheader("📄 Teks Asli")
    for i, kal in enumerate(sentences):
        st.write(f"{i+1}. {kal}")

    # 2. Preprocessing
    def preprocess_for_weight(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = tokenizer.tokenize(text)
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    preprocessed_sentences = [preprocess_for_weight(s) for s in sentences]

    # 3. TF-IDF + LSA
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

    n_components = min(1, len(sentences))
    lsa = TruncatedSVD(n_components=n_components)
    lsa_result = lsa.fit_transform(tfidf_matrix)

    # 4. Ringkasan = 50% kalimat teratas
    scores = lsa_result[:, 0]
    threshold = sorted(scores, reverse=True)[max(1, len(scores) // 2) - 1]
    selected_sentences = [sentences[i] for i, score in enumerate(scores) if score >= threshold]

    # 5. Output
    st.subheader("📌 Hasil Ringkasan (50% Kalimat Terpenting)")
    for i, kal in enumerate(selected_sentences):
        st.success(f"{i+1}. {kal}")
else:
    st.info("Masukkan teks terlebih dahulu dan klik 'Ringkas Sekarang' untuk melihat hasil.")
