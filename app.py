
import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
stemmer = StemmerFactory().create_stemmer()
stop_words = set(stopwords.words('indonesian'))

st.set_page_config(page_title="LSA Text Summarizer", layout="wide")

st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Comic Sans MS', cursive, sans-serif;
            background-color: #fff0f5;
        }
        .main {
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

st.title("💗 LSA Text Summarizer (Ringkasan Otomatis Bahasa Indonesia)")

uploaded_file = st.file_uploader("📄 Upload file Excel berisi teks", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📊 Tabel Data Awal")
    st.write(df.head())

    if 'Isi' not in df.columns:
        st.warning("Kolom 'Isi' tidak ditemukan dalam file.")
    else:
        def preprocessing(text):
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            tokens = word_tokenize(text)
            filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
            return ' '.join(filtered)

        df['preprocessed'] = df['Isi'].astype(str).apply(preprocessing)

        st.subheader("🔍 Hasil Preprocessing")
        st.write(df[['Isi', 'preprocessed']].head())

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['preprocessed'])
        lsa = TruncatedSVD(n_components=1)
        lsa_result = lsa.fit_transform(X)

        df['LSA_Score'] = lsa_result[:, 0]
        summary_index = df['LSA_Score'].idxmax()
        summary = df.loc[summary_index, 'Isi']

        st.subheader("📌 Ringkasan Otomatis")
        st.info(summary)

        st.subheader("📈 Visualisasi Skor LSA")
        fig, ax = plt.subplots()
        ax.plot(df['LSA_Score'], marker='o', color='#d63384')
        ax.set_title("Skor LSA per Dokumen")
        st.pyplot(fig)
