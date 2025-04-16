import pdfplumber
import re

def extract_email(text):
    pattern = r"[a-zA-Z0-9\._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match  = re.findall(pattern, text)
    return match[0] if match else None

def extract_phone(text):
    pattern = r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}" 
    match = re.findall(pattern, text)
    return match if match else None

def parse_resume(file_path):
    with pdfplumber.open(file_path) as pdf :
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text = text + page_text + "\n"

    email = extract_email(text)
    contact = extract_phone(text)

    data =  {"Email :" :email , "Contact : ": contact}

    for key,value in data.items():
        print(f"{key} : {value}")

    


file_path = "D://INTERN-PARAM//AI Screening//Resumes//Resume.pdf"
result = parse_resume(file_path)


    