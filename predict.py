"""
Prediction helpers shared by app.py and the command line.
Combines two things:
  1. ML model probability (TF-IDF + Logistic Regression)
  2. Rule-based red flags (registration fee, WhatsApp only, free email, etc.)
"""
import re

import joblib

MODEL_PATH = "model/model.pkl"
TFIDF_PATH = "model/tfidf.pkl"

# (pattern, human readable message)
RED_FLAGS = [
    (r"registration fee|security deposit|training fee|pay .* fee|refundable amount|processing fee",
     "Asks you to pay money (fee / deposit)"),
    (r"whatsapp only|contact (us )?on whatsapp|apply .* telegram|telegram",
     "Uses WhatsApp / Telegram for hiring"),
    (r"@(gmail|yahoo|hotmail|outlook)\.com",
     "Uses a free email address instead of a company email"),
    (r"no interview|without (any )?interview|direct joining",
     "Promises a job with no interview"),
    (r"no experience|no skills|no qualification|anyone can apply",
     "Needs no experience or skills but offers high pay"),
    (r"guaranteed|100%|earn (lakhs|rs\.? ?\d{4,})",
     "Unrealistic income promises"),
    (r"bank details|aadhaar|aadhar|pan card|otp",
     "Asks for sensitive personal or bank details early"),
    (r"urgent|limited seats|hurry|immediately",
     "Creates false urgency"),
]


def load_model():
    return joblib.load(MODEL_PATH), joblib.load(TFIDF_PATH)


def find_red_flags(text):
    found = []
    for pattern, message in RED_FLAGS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(message)
    return found


def check_job(text, model, tfidf):
    """Returns (label, fake_probability_percent, red_flags_list)."""
    vec = tfidf.transform([text])
    prob_fake = float(model.predict_proba(vec)[0][1]) * 100
    flags = find_red_flags(text)

    # Final decision: ML says fake OR 3+ red flags found
    is_fake = prob_fake >= 50 or len(flags) >= 3
    label = "FAKE" if is_fake else "REAL"
    return label, prob_fake, flags


if __name__ == "__main__":
    model, tfidf = load_model()
    sample = input("Paste a job description: ")
    label, prob, flags = check_job(sample, model, tfidf)
    print(f"Result: {label} (fake probability {prob:.1f}%)")
    for f in flags:
        print(" - Red flag:", f)
