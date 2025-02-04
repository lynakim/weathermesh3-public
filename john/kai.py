import requests
import json

def submit_application(name, email, resume_url, submission_url, notes=None):
    url = "https://windbornesystems.com/career_applications.json"
    
    payload = {
        "career_application": {
            "name": name,
            "email": email,
            "role": "Linux Systems and Networks Engineer",
            "resume_url": resume_url,
            "submission_url": submission_url,
            "notes": notes
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        print("Application submitted successfully!")
        print("Response:", response.text)
    else:
        print(f"Failed to submit application. Status code: {response.status_code}")
        print("Response:", response.text)

# Usage example
if __name__ == "__main__":
    name = "John Dean"
    email = "your.email@example.com"
    resume_url = "https://example.com/your_resume.pdf"
    submission_url = "https://example.com/your_coding_challenge"
    notes = "Do I really trust Kai Marshland?"

    submit_application(name, email, resume_url, submission_url, notes)