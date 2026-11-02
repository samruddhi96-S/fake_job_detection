# Fake Job Detection (Mini Project)

Detects whether a job posting is REAL or FAKE using TF-IDF + Logistic Regression
plus rule-based scam red flags. Web UI built with Streamlit.

## How to run in VS Code (Windows)

1. Unzip the folder, then in VS Code: File > Open Folder > select `fake_job_detection`.
2. Open the terminal: Terminal > New Terminal.
3. (Recommended) Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
   (Mac/Linux: `source venv/bin/activate`)
4. Install libraries:
   ```
   pip install -r requirements.txt
   ```
5. Train the model:
   ```
   python train.py
   ```
6. Start the app:
   ```
   streamlit run app.py
   ```
   The browser opens automatically at http://localhost:8501

## Using the real dataset (for final submission)

1. Download "Fake Job Postings" (EMSCAD) from Kaggle.
2. Save it as `data/fake_job_postings.csv`.
3. Run `python train.py` again. No code change is needed.

## Files

- `make_sample_data.py` : creates small demo data
- `train.py` : trains and compares 3 models, saves best model + confusion matrix
- `predict.py` : ML prediction + red-flag rules
- `app.py` : Streamlit web app
