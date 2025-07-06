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

# Konfigurasi halaman
st.set_page_config(page_title="LSA Summarizer", layout="wide")

# Tema CSS pastel
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
        .custom-textarea textarea {
            background-color: #ffffff;
            color: black;
            width: 400px !important;
            height: 100px !important;
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

# Input teks dan compression rate
with st.container():
    st.markdown("<div class='custom-textarea'>", unsafe_allow_html=True)
    input_text = st.text_area("Masukkan teks panjang untuk diringkas:")
    st.markdown("</div>", unsafe_allow_html=True)

compression_option = st.selectbox("Pilih Compression Rate", ["50%", "30%", "10%"])

if st.button("Ringkas Teks"):
    if input_text:
        # Pisahkan kalimat
        sentences = re.split(r'(?<=[.!?])\s+', input_text)
        sentences = [s.strip() for s in sentences if s.strip()]

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
        scores = svd.fit_transform(X)[:, 0]

        # Tentukan jumlah kalimat berdasarkan compression rate
        total_sentences = len(sentences)
        if compression_option == "50%":
            n = max(1, int(total_sentences * 0.5))
        elif compression_option == "30%":
            n = max(1, int(total_sentences * 0.3))
        else:
            n = max(1, int(total_sentences * 0.1))

        # Pilih kalimat berdasarkan skor tertinggi
        top_indices = np.argsort(-scores)[:n]
        summary = " ".join([sentences[i] for i in sorted(top_indices)])

        # Tampilkan hasil
        st.subheader(f"Ringkasan Teks (Compression Rate {compression_option})")
        st.markdown(f"**Jumlah Kalimat Asli:** {total_sentences}")
        st.markdown(f"**Jumlah Kalimat Setelah Ringkasan:** {n}")
        st.markdown(
            f"<div style='background-color:#fce4ec; color:#333; padding: 1rem; border-radius: 10px; font-size: 16px;'>{summary}</div>",
            unsafe_allow_html=True)
    else:
        st.warning("Teks tidak boleh kosong.")
