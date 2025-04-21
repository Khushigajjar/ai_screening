import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet



def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
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
        # Direct match
        if job_word in resume_keywords_set:
            matched_keywords.append(job_word)
        else:
            # Synonym match
            job_word_synonyms = get_synonyms(job_word)
            if job_word_synonyms & resume_keywords_set:
                matched_keywords.append(job_word)
    
    return {
        "Matched Keywords": matched_keywords,
        "Similarity Score": similarity_score * 100
    }


job_description = '''3-7 years’ experience in building backend services using Python.
Good knowledge on cloud platforms like AWS and Google Cloud 
Bachelor’s degree in computer science or engineering discipline
Experience with MySQL and/or PostgreSQL databases.
Strong coding and debugging skills with experience in backend technologies.
Strong problem-solving skills and attention to detail.
Exposure in building micro services applications architecture and with hands on experience in design and development/ coding. 
Knowledge with development tools such as IntelliJ, Visual Studio Code, GitLab, BitBucket.
Strong inter-personal communication and collaboration skills. The candidate must be able to work independently & must possess strong communication skills.
A good team player, high level of personal commitment & 'can do' attitude.
Demonstrable ownership, commitment and willingness to learn new technologies and frameworks'''

resume_text = "Machine Learning Data Science python Mysql databases."

result = compare_resume_with_job(job_description, resume_text)

print("Matched Keywords:", result['Matched Keywords'])
print("Similarity Score:", result['Similarity Score'])