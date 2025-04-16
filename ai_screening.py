import pdfplumber
import re
import os
import phonenumbers

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

    