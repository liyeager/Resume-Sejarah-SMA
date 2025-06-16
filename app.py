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

st.set_page_config(page_title="LSA Summarizer - AOT Theme", layout="wide")
st.markdown("""
    <style>
        body {
            background-image: url('https://i.ibb.co/X2PG7yQ/aot-bg.jpg');
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }
        .reportview-container .main .block-container{
            padding-top: 2rem;
            padding-bottom: 2rem;
            background-color: rgba(0, 0, 0, 0.6);
            border-radius: 12px;
        }
        .stButton>button {
            color: white;
            background-color: #223f57;
            border-radius: 8px;
        }
        .stTextArea textarea {
            background-color: #f2f2f2;
            color: black;
        }
        header, footer, .st-emotion-cache-1v0mbdj, .st-emotion-cache-z5fcl4 {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

st.title("LSA Summarizer - Attack on Titan Edition")

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
        st.subheader("Hasil Pemisahan Kalimat")
        for s in sentences:
            st.markdown(f"- {s}")

        st.subheader("Preprocessing Tiap Kalimat")
        for p in preprocessed_sentences:
            st.markdown(f"- {p}")

        st.subheader("Matriks TF-IDF")
        st.dataframe(tfidf_df, use_container_width=True)

        st.subheader("Skor SVD per Kalimat")
        st.dataframe(svd_df, use_container_width=True)

        st.subheader(f"Ringkasan Teks (Top {n} kalimat / 10%)")
        st.markdown(f"<div style='background-color:#1c1c1c; color:white; padding: 1rem; border-radius: 10px;'>{summary}</div>", unsafe_allow_html=True)

    else:
        st.warning("Teks tidak boleh kosong.")
