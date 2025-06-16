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

st.set_page_config(page_title="LSA Summarizer", layout="wide", page_icon="📘")
st.markdown("""
    <style>
        body {
            background-color: #eaf4fb;
        }
        .reportview-container .main .block-container{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            color: white;
            background-color: #3c91e6;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📘 LSA Text Summarizer (10% Compression Rate)")

input_text = st.text_area("Masukkan teks panjang untuk diringkas:", height=400)

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
        features = vectorizer.get_feature_names_out()

        tfidf_df = pd.DataFrame(X.toarray(), columns=features)
        tfidf_df.insert(0, "Kalimat Asli", sentences)

        # SVD
        svd = TruncatedSVD(n_components=1, random_state=42)
        X_svd = svd.fit_transform(X)
        scores = X_svd[:, 0]
        svd_df = pd.DataFrame({"Kalimat": sentences, "Skor SVD": scores})

        # Ringkasan 10%
        n = max(1, int(len(sentences) * 0.1))
        top_indices = np.argsort(-scores)[:n]
        summary = " ".join([sentences[i] for i in sorted(top_indices)])

        # Output
        st.subheader("📌 Hasil Pemisahan Kalimat")
        for i, s in enumerate(sentences, 1):
            st.markdown(f"**{i}.** {s}")

        st.subheader("🔧 Preprocessing Tiap Kalimat")
        for i, p in enumerate(preprocessed_sentences, 1):
            st.markdown(f"**{i}.** {p}")

        st.subheader("📊 Matriks TF-IDF")
        st.dataframe(tfidf_df, use_container_width=True)

        st.subheader("📈 Skor SVD per Kalimat")
        st.dataframe(svd_df, use_container_width=True)

        st.subheader(f"📄 Ringkasan Teks (Top {n} kalimat / 10%)")
        st.markdown(f"<div style='background-color:#d9ecff; padding: 1rem; border-radius: 10px;'>{summary}</div>", unsafe_allow_html=True)

    else:
        st.warning("Teks tidak boleh kosong.")
