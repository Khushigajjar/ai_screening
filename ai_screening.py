import pdfplumber
import re
import os
import phonenumbers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 

def extract_email(text):
    pattern = r"[a-zA-Z0-9\._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match  = re.findall(pattern, text)
    return match[0] if match else None

def extract_phone(text):
    phone_numbers = []
    for match in phonenumbers.PhoneNumberMatcher(text, "IN"):
        number = match.number
        if phonenumbers.is_valid_number(number):
            formatted = phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
            phone_numbers.append(formatted)
    return phone_numbers if phone_numbers else None

def extract_skills(text, job_description):

    resume_vectorizer = TfidfVectorizer(stop_words="english")
    resume_matrix = resume_vectorizer.fit_transform([text])
    resume_keywords = resume_vectorizer.get_feature_names_out()

    document = [job_description.lower(), ' '.join(resume_keywords)]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(document)
    similarity = cosine_similarity(matrix[:1], matrix[1:2])[0][0]

    job_tokens = set(job_description.lower().split())
    resume_tokens = set(resume_keywords)
    matched_keywords = job_tokens.intersection(resume_tokens)

    if matched_keywords:
        return resume_keywords, similarity, list(matched_keywords)
    else:
        return resume_keywords, similarity, ["No Matching Skills"]


def parse_resume(file_path):
    with pdfplumber.open(file_path) as pdf :
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=2)
            if page_text:
                text = text + page_text + "\n"

                print(text)

    email = extract_email(text)
    contact = extract_phone(text)
    job_description = "Python, Machine Learning, and Data Analysis"


    similarity_score, extracted_skills, res_keywords = extract_skills(text, job_description)



    data = {"file_name" : os.path.basename(file_path), 
            "Email" : email , 
            "Contact" : contact,
            "Similarity_score" : similarity_score,
            "Skills" : extracted_skills,
            "Keywords" : res_keywords}

    return data

folder = "./Resumes"
parsed_data = []

for file_name in os.listdir(folder):
    file_path = os.path.join(folder, file_name)
    result = parse_resume(file_path)
    parsed_data.append(result)


for i in parsed_data:
    print(f"File: {i['file_name']}")
    print(f"Email : {i['Email']}")
    print(f"Contact : {i['Contact']}")
    print(f"Similarity_score : {i["Similarity_score"]*100:.2f}%")
    print(f"Skills : {i["Skills"]}")
    print(f"res_keywords: {i["res_keywords"]}")
    print("-" * 40)

    