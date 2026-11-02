"""
Creates a small SAMPLE dataset so the project runs instantly.
For the final submission, download the real EMSCAD dataset from Kaggle
("Fake Job Postings") and save it as data/fake_job_postings.csv
(it has the same column names, so no code change is needed).
"""
import os
import random

import pandas as pd

random.seed(42)

REAL_TITLES = ["Software Engineer", "Data Analyst", "Marketing Executive", "HR Manager",
               "Accountant", "Sales Associate", "Graphic Designer", "Customer Support Executive",
               "Python Developer", "Business Analyst", "Operations Manager", "Content Writer"]
FAKE_TITLES = ["Work From Home Data Entry", "Earn Money Online Typing", "Easy Part Time Job",
               "Urgent Hiring Online Assistant", "Home Based Packing Job", "Online Form Filling Job",
               "Instant Joining Receptionist", "Get Paid Daily Simple Task"]

REAL_COMPANY = [
    "We are an established technology company with offices in several cities and over ten years of experience.",
    "Our firm provides consulting and software services to banks, retailers and healthcare clients.",
    "A growing product company focused on cloud solutions with a diverse and supportive team.",
]
REAL_DESC = [
    "You will work with the engineering team to design, build and test features. Participate in code reviews and sprint planning.",
    "Responsibilities include analysing business data, preparing weekly reports and presenting insights to stakeholders.",
    "The role involves managing client relationships, coordinating with internal teams and meeting quarterly targets.",
    "You will be responsible for drafting content, editing articles and working with the design team on campaigns.",
]
REAL_REQ = [
    "Bachelor's degree in a relevant field. 2 years of experience. Good communication skills. Interview rounds will be held at our office.",
    "Strong knowledge of SQL, Excel and reporting tools. Ability to work in a team. Formal interview process with HR and technical rounds.",
    "Experience with Python or Java. Understanding of data structures. Two technical interviews and one HR discussion.",
]
REAL_BEN = ["Health insurance, paid leave, learning budget and flexible hours.",
            "Provident fund, annual bonus and career growth opportunities.",
            "Competitive salary, medical cover and team events."]

FAKE_COMPANY = ["", "", "Leading international company hiring urgently.", "Top MNC. Limited seats only."]
FAKE_DESC = [
    "Earn Rs 50000 per week working from home. No experience needed. Just pay a small registration fee to start. Contact us on WhatsApp now.",
    "Easy typing job. Earn money daily without any interview. Send your bank details and Aadhaar copy to get joining letter. Limited seats!!!",
    "Urgent requirement. Get paid daily for simple tasks. Security deposit required for the work kit. Apply immediately on Telegram.",
    "100% guaranteed income, no skills required. Pay training fee and start earning lakhs from home. Contact hr.jobs.india@gmail.com",
    "Direct joining, no interview, no qualification needed. Transfer refundable amount to confirm your seat. WhatsApp only.",
]
FAKE_REQ = ["No experience. Just a smartphone.", "Anyone can apply. Pay registration fee.", "", "Only 10th pass. Send documents on WhatsApp."]
FAKE_BEN = ["Unlimited earning!!!", "Earn lakhs per month", "", "Daily payment guaranteed"]


def make_row(fake):
    if fake:
        return {
            "title": random.choice(FAKE_TITLES),
            "company_profile": random.choice(FAKE_COMPANY),
            "description": random.choice(FAKE_DESC),
            "requirements": random.choice(FAKE_REQ),
            "benefits": random.choice(FAKE_BEN),
            "fraudulent": 1,
        }
    return {
        "title": random.choice(REAL_TITLES),
        "company_profile": random.choice(REAL_COMPANY),
        "description": random.choice(REAL_DESC),
        "requirements": random.choice(REAL_REQ),
        "benefits": random.choice(REAL_BEN),
        "fraudulent": 0,
    }


if __name__ == "__main__":
    rows = [make_row(False) for _ in range(850)] + [make_row(True) for _ in range(150)]
    random.shuffle(rows)
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(rows).to_csv("data/sample_jobs.csv", index=False)
    print("Sample data created: data/sample_jobs.csv (1000 rows)")
