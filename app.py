import streamlit as st
import string
import numpy as np
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Inisialisasi tools lokal
stemmer = StemmerFactory().create_stemmer()
stop_words = set(StopWordRemoverFactory().get_stop_words())
tokenizer = RegexpTokenizer(r'\w+')

# Streamlit Setup
st.set_page_config(page_title="Coquette LSA Summarizer", layout="wide")

# 🎀 CSS: Tema Coquette Pastel
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Comic Sans MS', cursive, sans-serif;
            background-color: #fff0f5;
            color: #6a1b9a;
        }
        h1, h2, h3 {
            color: #c2185b;
        }
        .stTextArea textarea {
            background-color: #ffe4ec;
            border: 1.5px solid #e91e63;
            color: #4a148c;
            border-radius: 12px;
        }
        .stButton>button {
            background-color: #ffd6e8;
            color: #b71c1c;
            font-weight: bold;
            border-radius: 20px;
            border: 2px solid #f06292;
            transition: 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #f8bbd0;
            color: #880e4f;
        }
        .stAlert {
            border-radius: 16px;
            background-color: #fce4ec;
            border-left: 6px solid #f06292;
        }
    </style>
""", unsafe_allow_html=True)

# Judul
st.title("💗 Coquette LSA Text Summarizer")

# Input Teks
input_text = st.text_area("📝 Tulis atau tempel teks yang ingin diringkas:")

if st.button("💖 Ringkas Sekarang") and input_text.strip():
    # 1. Split kalimat
    sentences = re.split(r'(?<=[.!?])\s+', input_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    st.subheader("📄 Teks Asli (Paragraf)")
    st.write(" ".join(sentences))

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

    # 4. Ringkasan 50%
    scores = lsa_result[:, 0]
    threshold = sorted(scores, reverse=True)[max(1, len(scores) // 2) - 1]
    selected_sentences = [sentences[i] for i, score in enumerate(scores) if score >= threshold]

    # 5. Tampilkan hasil
    st.subheader("🌷 Hasil Ringkasan (Paragraf, 50% Terpenting)")
    st.success(" ".join(selected_sentences))

else:
    st.info("Masukkan teks terlebih dahulu dan klik '💖 Ringkas Sekarang'.")
