import streamlit as st
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="LSA Text Summarizer", layout="wide")

# Styling
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

# Input teks dari user
input_text = st.text_area("📝 Masukkan teks untuk diringkas:")

if st.button("🔍 Ringkas Sekarang") and input_text.strip():
    # Pemrosesan
    sentences = input_text.replace('\n', ' ').split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    st.subheader("📄 Teks Asli")
    for i, kal in enumerate(sentences):
        st.write(f"{i+1}. {kal}")

    # Preprocessing untuk pembobotan
    stemmer = StemmerFactory().create_stemmer()
    stop_words = set(stopwords.words('indonesian'))

    def preprocess_for_weight(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    # Preprocess tiap kalimat
    preprocessed_sentences = [preprocess_for_weight(s) for s in sentences]

    # TF-IDF dan LSA
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

    n_components = min(1, len(sentences))  # satu komponen karena ingin ambil skor utama
    lsa = TruncatedSVD(n_components=n_components)
    lsa_result = lsa.fit_transform(tfidf_matrix)

    # Ambil 50% kalimat dengan skor tertinggi
    scores = lsa_result[:, 0]
    threshold = sorted(scores, reverse=True)[max(1, len(scores) // 2) - 1]
    selected_sentences = [sentences[i] for i, score in enumerate(scores) if score >= threshold]

    # Tampilkan hasil ringkasan
    st.subheader("📌 Hasil Ringkasan (50% Kalimat Terpenting)")
    for i, kal in enumerate(selected_sentences):
        st.success(f"{i+1}. {kal}")
else:
    st.info("Masukkan teks terlebih dahulu dan klik 'Ringkas Sekarang' untuk melihat hasil.")
