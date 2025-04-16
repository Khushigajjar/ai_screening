import pdfplumber
import re
import os

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

    data = {"file_name" : os.path.basename(file_path), 
            "Email" : email , 
            "Contact" : contact}

    return data

    


folder = "./Resumes"
parsed_data = []

for file_name in os.listdir(folder):
    file_path = os.path.join(folder, file_name)
    result = parse_resume(file_path)
    parsed_data.append(result)


for i in parsed_data:
    print(f"File: {i['file_name']}")
    print(f"Email : {i["Email"]}")
    print(f"Contact : {i["Contact"]}")
    print("-"*40)

    