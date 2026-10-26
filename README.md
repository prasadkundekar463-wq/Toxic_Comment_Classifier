# Toxic Comment Detection Using Machine Learning

## Files
- `app.py`: Streamlit demo app.
- `requirements.txt`: Python packages needed to run the app.
- `Toxic_Comment_Detection_Project_Report.docx`: project report template/content.
- `Toxic_Comment_Detection_Presentation.pptx`: presentation.

## Required model files
Keep these files in the same folder as `app.py`:
- toxic_nontoxic_model.pkl
- toxic_nontoxic_vectorizer.pkl
- severe_toxic_model.pkl / severe_toxic_vectorizer.pkl
- obscene_model.pkl / obscene_vectorizer.pkl
- threat_model.pkl / threat_vectorizer.pkl
- insult_model.pkl / insult_vectorizer.pkl
- identity_hate_model.pkl / identity_hate_vectorizer.pkl

## Run locally
```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Important: the app expects the saved models to be compatible with the vectorizers and scikit-learn version used to train them. Do not publish private information or datasets without permission. This is an educational prototype and predictions can be wrong.
