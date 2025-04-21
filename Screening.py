import re
import string
import fitz 
import os
from nltk.corpus import stopwords, wordnet
import phonenumbers  # type: ignore
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

    vectorizer = TfidfVectorizer()
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

def compare_resume_with_job(job_description, resume_text):
    job_keywords = extract_keywords(job_description)
    resume_keywords = extract_keywords(resume_text)
    similarity_score = calculate_similarity(job_description, resume_text)
    
    matched_keywords = []
    resume_keywords_set = set(resume_keywords)
    
    for job_word in job_keywords:
        if job_word in resume_keywords_set:
            matched_keywords.append(job_word)
        else:
            job_word_synonyms = get_synonyms(job_word)
            if job_word_synonyms & resume_keywords_set:
                matched_keywords.append(job_word)
    
    matched_keywords = sorted(list(set(matched_keywords)))  
    return {
        "Matched Keywords": matched_keywords,
        "Similarity Score": similarity_score * 100
    }

if __name__ == "__main__":
    print("Enter the Job Description : \n")
    job_description_lines = []
    while True:
        line = input()
        if line == "":
            break
        job_description_lines.append(line)
    job_description = "\n".join(job_description_lines)

    resumes_folder = r'D:\INTERN-PARAM\AI Screening\Resumes'
    resumes_list = [f for f in os.listdir(resumes_folder) if f.endswith('.pdf')]

    if not resumes_list:
        print("No resumes found in the folder!")
        exit()
    all_results = []

    for file_name in resumes_list:
        resume_path = os.path.join(resumes_folder, file_name)
        resume_text = extract_text_from_pdf(resume_path)
        result = compare_resume_with_job(job_description, resume_text)
        email = extract_email(resume_text)
        phones = extract_phone(resume_text)

        all_results.append((file_name, result['Similarity Score'], result['Matched Keywords'], email, phones))

all_results = sorted(all_results, key=lambda x: x[1], reverse=True)

for idx, (file_name, score, keywords, email, phones) in enumerate(all_results, 1):
    print(f"{idx}. {file_name}")
    print(f"   Similarity Score: {round(score, 2)}%")
    print(f"   Matched Keywords: {keywords}")
    print(f"   Email: {email}")
    print(f"   Phone Numbers: {phones}\n")
