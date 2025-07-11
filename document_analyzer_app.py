import streamlit as st
import pandas as pd
import docx2txt
import PyPDF2
import io
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Page config
st.set_page_config(page_title="Document Analyzer", layout="wide")
st.title("📄 Document Analyzer")
st.markdown("Upload a PDF or Word document to extract text and get quick insights.")

# Upload
uploaded_file = st.file_uploader("Upload your document (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    text = ""

    # Extract text
    if uploaded_file.name.endswith(".docx"):
        text = docx2txt.process(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text()

    if text:
        st.subheader("📜 Extracted Text")
        st.text_area("Raw Text", text, height=200)

        st.subheader("🔍 Basic Stats")
        words = text.split()
        filtered_words = [w.lower() for w in words if w.lower() not in stop_words and w.isalpha()]
        st.write(f"**Total Words:** {len(words)}")
        st.write(f"**Filtered Words (no stopwords):** {len(filtered_words)}")

        st.subheader("☁️ Word Cloud")
        wordcloud = WordCloud(width=800, height=300, background_color="white").generate(" ".join(filtered_words))
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        st.subheader("📊 Top 10 Words")
        vectorizer = CountVectorizer(stop_words='english')
        word_freq = vectorizer.fit_transform([text])
        word_counts = pd.DataFrame(word_freq.toarray(), columns=vectorizer.get_feature_names_out()).T
        word_counts.columns = ["count"]
        top_words = word_counts.sort_values(by="count", ascending=False).head(10)
        st.bar_chart(top_words)

    else:
        st.error("❌ Could not extract text from the file.")
