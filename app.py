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
st.set_page_config(page_title="LSA Text Summarizer", layout="wide")

# Tambahan gaya coquette
st.markdown("""
    <style>
        html, body, [class^="css"]  {
            font-family: 'Comic Sans MS', cursive, sans-serif;
            background-color: #fce4ec;
            color: #880e4f;
        }
        h1, h2, h3, .stMarkdown {
            color: #ad1457;
        }
        .stButton>button {
            background-color: #f8bbd0;
            color: #880e4f;
            border: 1px solid #ad1457;
            border-radius: 12px;
            font-weight: bold;
        }
        .stTextArea textarea {
            background-color: #fff0f5;
            border: 1px solid #f48fb1;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💗 LSA Text Summarizer - Coquette Style")

# Input dari pengguna
input_text = st.text_area("📝 Masukkan teks untuk diringkas:")

if st.button("🔍 Ringkas Sekarang") and input_text.strip():
    # 1. Split jadi kalimat manual
    sentences = re.split(r'(?<=[.!?])\s+', input_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    st.subheader("📄 Teks Asli")
    st.markdown(f"""<div style='background-color:#fce4ec; padding:10px; border-radius:10px'>{input_text}</div>""", unsafe_allow_html=True)

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

    # 5. Output sebagai paragraf utuh
    summary_paragraph = " ".join(selected_sentences)

    st.subheader("📌 Hasil Ringkasan (50% Kalimat Terpenting)")
    st.markdown(f"""<div style='background-color:#f8bbd0; padding:15px; border-radius:15px; font-size:16px;'>
        {summary_paragraph}
    </div>""", unsafe_allow_html=True)

else:
    st.info("Masukkan teks terlebih dahulu dan klik 'Ringkas Sekarang' untuk melihat hasil.")
