import pdfplumber
import re
import os
import string
import phonenumbers  # type: ignore
from nltk.corpus import stopwords
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
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    tokens = [lemmatizer.lemmatize(word, pos='n') for word in tokens]
    return " ".join(tokens)


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

    print("Preprocessed Job Description:", job_description)
    print("Preprocessed Resume Text:", resume_text)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_description, resume_text])

    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity_score


def extract_keywords(text):
    text = preprocess_text(text)
    keywords = text.split()
    return keywords


def compare_resume_with_job(job_description, resume_text):
    job_keywords = extract_keywords(job_description)
    resume_keywords = extract_keywords(resume_text)
    similarity_score = calculate_similarity(job_description, resume_text)
    matched_keywords = list(set(job_keywords).intersection(resume_keywords))
    return {
        "Matched Keywords": matched_keywords,
        "Similarity Score": similarity_score * 100
    }


def parse_resume(file_path, job_description):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=2)
            if page_text:
                text += page_text + "\n"

    email = extract_email(text)
    contact = extract_phone(text)
    result = compare_resume_with_job(job_description, text)

    data = {
        "file_name": os.path.basename(file_path),
        "Email": email,
        "Contact": contact,
        "Similarity Score (%)": result['Similarity Score'],
        "Matched Keywords": result['Matched Keywords']
    }

    return data


# Main script
folder = "./Resumes"
parsed_data = []

job_description = input("Enter Job Description: ")

for file_name in os.listdir(folder):
    file_path = os.path.join(folder, file_name)
    if file_path.endswith(".pdf"):
        result = parse_resume(file_path, job_description)
        parsed_data.append(result)

for i in parsed_data:
    print(f"File: {i['file_name']}")
    print(f"Email: {i['Email']}")
    print(f"Contact: {i['Contact']}")
    print(f"Similarity Score: {i['Similarity Score (%)']:.2f}%")
    print(f"Matched Keywords: {i['Matched Keywords']}")
    print("-" * 50)
