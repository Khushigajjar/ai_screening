import re
import string
import fitz
import os
import zipfile
import tempfile
import pandas as pd
import streamlit as st
import phonenumbers # type: ignore
import spacy
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
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
    doc = nlp(text)
    keywords = []

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "VERB"]:
            if not token.is_stop and not token.is_punct and len(token.text) > 2:
                keywords.append(token.lemma_.lower())

    return list(set(keywords))



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
    intersection_score = (len(intersection_keywords) / len(job_keywords)) * 100 if job_keywords else 0

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
        "job_keywords" : job_keywords,
        "resume_keywords" : resume_keywords,
        "intersection_score" : intersection_score,
        "similarity_score" : similarity_score,
        "Match Percentage (%)": round(combined_score, 2),
    }

def main():
    st.set_page_config(page_title="Resume Screening", layout="wide")
    st.title("Happy Resume Screening :)")

    job_description = st.text_area("Enter the Job Description:", height=200)

    uploaded_file = st.file_uploader("Upload Resumes (ZIP or single PDF):", type=["zip", "pdf"])

    if job_description and uploaded_file:
        results = []

        with tempfile.TemporaryDirectory() as temp_dir:
            file_name = uploaded_file.name

            if file_name.endswith('.zip'):
                zip_path = os.path.join(temp_dir, "resumes.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                if zipfile.is_zipfile(zip_path):
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
                                    "similarity_score" : result["similarity_score"],
                                    "intersection_score" : result["intersection_score"],
                                    "Match Percentage (%)": result['Match Percentage (%)'],
                                    "Matched Keywords": ", ".join(result['Matched Keywords']),
                                    "job_keywords": ", ".join(result['job_keywords']),
                                    "resume_keywords": ", ".join(result['resume_keywords'])
                                })
                else:
                    st.error("Uploaded ZIP file is not valid.")

            elif file_name.endswith('.pdf'):
                pdf_path = os.path.join(temp_dir, file_name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                resume_text = extract_text_from_pdf(pdf_path)
                result = compare_resume_with_job(job_description, resume_text)
                email = extract_email(resume_text)
                phones = extract_phone(resume_text)

                results.append({
                    "File Name": file_name,
                    "Email": email,
                    "Phone Numbers": ", ".join(phones) if phones else None,
                    "intersection_score" : result["intersection_score"],
                    "similarity_score" : result["similarity_score"],
                    "job_keywords": ", ".join(result['job_keywords']),
                    "resume_keywords": ", ".join(result['resume_keywords']),
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
