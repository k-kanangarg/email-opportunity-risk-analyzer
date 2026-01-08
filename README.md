# Email Opportunity & Risk Analyzer
# Email Opportunity & Risk Analyzer

A risk-aware email analysis tool that identifies internship and career opportunities, estimates spam likelihood, and flags potentially paid or misleading programs.

## What it does
- Estimates spam probability using classical NLP and machine learning
- Detects internship and career opportunity emails
- Flags high-risk or pay-to-play programs instead of blindly filtering them
- Extracts application links and relevant signals for user review

## Why this exists
Many internship opportunities arrive mixed with spam or misleading training programs. This project focuses on responsible decision support by separating spam detection from opportunity risk assessment to reduce user harm.

## Tech Stack
- Python
- Scikit-learn (TF-IDF, Multinomial Naive Bayes)
- Streamlit
- Rule-based NLP
