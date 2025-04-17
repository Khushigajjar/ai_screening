import spacy
import pdfplumber

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Extract candidate skills using NER and noun chunks
def extract_skills(text):
    doc = nlp(text)
    skills = set()

    # Noun chunks like "data analysis", "machine learning"
    for chunk in doc.noun_chunks:
        if len(chunk.text) > 2:
            skills.add(chunk.text.strip().lower())

    # Named entities (sometimes tools/technologies are tagged)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]:
            skills.add(ent.text.strip().lower())

    return list(skills)

# Run
file_path = "./Resumes/resume.pdf"  # Change to your file
resume_text = extract_text_from_pdf(file_path)
skills_found = extract_skills(resume_text)

print("Extracted Skills:", skills_found)
