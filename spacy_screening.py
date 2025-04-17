import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer, WordNetLemmatizer

def preprocess_text(text):

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # stemmer = PorterStemmer()
    # tokens = [stemmer.stem(word) for word in tokens]
    lemmatizer = WordNetLemmatizer()
    #tokens = []
    return " ".join(tokens)

def calculate_similarity(job_description, resume_text):
    job_description = preprocess_text(job_description)
    resume_text = preprocess_text(resume_text)

    print("preprocessed text from job : ", job_description)
    print("From resume : ", resume_text)

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
        "Similarity Score": similarity_score*100
        }

job_description = " i want students from Python, Machine Learning, Data Science"
resume_text = "Machine Learning Data Science python."

result = compare_resume_with_job(job_description, resume_text)

print("Matched Keywords:", result['Matched Keywords'])
print("Similarity Score:", result['Similarity Score'])