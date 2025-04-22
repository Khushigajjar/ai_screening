import re
import string
import fitz
import os
import zipfile
import tempfile
import pandas as pd
import streamlit as st
import phonenumbers # type:ignore
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_email(text):
    pattern = r"[a-zA-Z0-9\._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.findall(pattern, text)
    return match[0] if match else None


def extract_phone(text):
    phone_numbers = []
    for match in phonenumbers.PhoneNumberMatcher(text, "IN"):
        number = match.number
        if phonenumbers.is_valid_number(number):
            formatted = phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
            phone_numbers.append(formatted)
    return phone_numbers if phone_numbers else None


def calculate_similarity(job_description, resume_text):
    job_description = preprocess_text(job_description)
    resume_text = preprocess_text(resume_text)

    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform([job_description, resume_text])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return similarity_score


def extract_keywords(text):
    text = preprocess_text(text)
    keywords = text.split()
    return keywords


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def combined_Score(job_description, resume_text):
    job_keywords = extract_keywords(job_description)
    resume_keywords = extract_keywords(resume_text)
    resume_keywords_set = set(resume_keywords)

    intersection_keywords = set(job_keywords) & resume_keywords_set
    intersection_score = (len(intersection_keywords) / len(job_keywords)) * 100 

    similarity_score = calculate_similarity(job_description, resume_text) * 100
    combined_score = (intersection_score * 0.7) + (similarity_score * 0.3)

    return intersection_keywords, intersection_score, similarity_score, combined_score



def compare_resume_with_job(job_description, resume_text):
    job_keywords = extract_keywords(job_description)
    resume_keywords = extract_keywords(resume_text)

    intersection_keywords, intersection_score, similarity_score, combined_score = combined_Score(job_description, resume_text)

    matched_keywords = list(intersection_keywords)


    for job_word in job_keywords:
        if job_word not in intersection_keywords:
            job_word_synonyms = get_synonyms(job_word)
            if job_word_synonyms & set(resume_keywords):
                matched_keywords.append(job_word)

    matched_keywords = sorted(list(set(matched_keywords)))

    return {
        "Matched Keywords": matched_keywords,
        "Match Percentage (%)": round(combined_score, 2),
    }

def main():
    st.set_page_config(page_title="Resume Screening", layout="wide")
    st.title("Happy Resume Screening :)")

    job_description = st.text_area("Enter the Job Description:", height=200)

    uploaded_zip = st.file_uploader("Upload a ZIP containing Resumes (PDFs):", type="zip")

    if job_description and uploaded_zip:
        results = []
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "resumes.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getvalue())

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.pdf'):
                        pdf_path = os.path.join(root, file)
                        resume_text = extract_text_from_pdf(pdf_path)
                        result = compare_resume_with_job(job_description, resume_text)
                        email = extract_email(resume_text)
                        phones = extract_phone(resume_text)

                        results.append({
                        "File Name": file,
                        "Email": email,
                        "Phone Numbers": ", ".join(phones) if phones else None,
                        "Match Percentage (%)": result['Match Percentage (%)'],
                        "Matched Keywords": ", ".join(result['Matched Keywords'])
                        })


        results.sort(key=lambda x: x["Match Percentage (%)"], reverse=True)

        if results:
            st.subheader("Matching Results")
            df = pd.DataFrame(results)
            st.dataframe(df)

if __name__ == "__main__":
    main()