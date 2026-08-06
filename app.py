# ============================================================
# FAKE JOB POSTING DETECTION DASHBOARD
# Project:
# A Hybrid Data Mining Framework for Fake Job Posting Detection
# ============================================================

from pathlib import Path
import html
import re
import string

import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PROJECT FILE PATHS
# ============================================================
# app.py must be in the main project folder.

MODEL_PATH = Path("models/best_model.pkl")

TFIDF_PATH = Path(
    "data/processed/tfidf_vectorizer.pkl"
)

ENCODER_PATH = Path(
    "data/processed/onehot_encoder.pkl"
)

PROCESSED_DATA_PATH = Path(
    "data/processed/processed_jobs.csv"
)

MODEL_RESULTS_PATH = Path(
    "results/model_comparison.csv"
)

MISCLASSIFIED_PATH = Path(
    "results/misclassified_samples.csv"
)


# ============================================================
# 3. FEATURE COLUMNS
# ============================================================

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

CATEGORICAL_COLUMNS = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

NUMERICAL_COLUMNS = [
    "text_length",
    "word_count",
    "avg_word_length",
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "salary_range_missing",
    "department_missing",
    "company_profile_missing",
    "requirements_missing",
    "benefits_missing",
]


# ============================================================
# 4. MODERN DARK DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(34, 197, 94, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(59, 130, 246, 0.10),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #040914 0%,
                #07111f 50%,
                #091827 100%
            );
        color: #f8fafc;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Hide Streamlit decoration */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-family: Inter, Arial, sans-serif;
    }

    p, li, label {
        color: #dbeafe;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1.03rem;
        line-height: 1.7;
        margin-bottom: 1.6rem;
        max-width: 900px;
    }

    .section-label {
        color: #4ade80;
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #07111f 0%,
                #0a1726 100%
            );
        border-right: 1px solid rgba(148, 163, 184, 0.14);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.4rem;
    }

    div[role="radiogroup"] label {
        background-color: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(148, 163, 184, 0.10);
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 7px;
        transition: all 0.2s ease;
    }

    div[role="radiogroup"] label:hover {
        background-color: rgba(34, 197, 94, 0.08);
        border-color: rgba(34, 197, 94, 0.32);
        transform: translateX(3px);
    }

    /* Form card */
    div[data-testid="stForm"] {
        background:
            linear-gradient(
                145deg,
                rgba(15, 29, 46, 0.97),
                rgba(9, 21, 35, 0.97)
            );
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 20px;
        padding: 1.6rem;
        box-shadow:
            0 18px 45px rgba(0, 0, 0, 0.28);
    }

    /* Inputs */
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #091522 !important;
        color: #f8fafc !important;
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        border-radius: 11px !important;
    }

    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: #22c55e !important;
        box-shadow:
            0 0 0 2px rgba(34, 197, 94, 0.14) !important;
    }

    .stTextInput label,
    .stTextArea label,
    .stSelectbox label,
    .stCheckbox label {
        color: #dbeafe !important;
        font-weight: 600;
    }

    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: #64748b !important;
    }

    /* Buttons */
    .stButton > button,
    .stFormSubmitButton > button {
        width: 100%;
        min-height: 48px;
        border: none;
        border-radius: 12px;
        background:
            linear-gradient(
                135deg,
                #16a34a,
                #22c55e
            );
        color: white;
        font-size: 1rem;
        font-weight: 750;
        box-shadow:
            0 10px 24px rgba(34, 197, 94, 0.20);
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background:
            linear-gradient(
                135deg,
                #22c55e,
                #4ade80
            );
        transform: translateY(-2px);
        box-shadow:
            0 14px 28px rgba(34, 197, 94, 0.27);
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(16, 30, 48, 0.98),
                rgba(9, 21, 35, 0.98)
            );
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 16px;
        padding: 18px;
        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.22);
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-weight: 750;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: rgba(15, 29, 46, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 14px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] summary {
        color: #e2e8f0 !important;
        font-weight: 650;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 14px;
        overflow: hidden;
    }

    /* Custom cards */
    .decision-card {
        padding: 22px;
        border-radius: 16px;
        margin-top: 12px;
        margin-bottom: 18px;
        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.22);
    }

    .decision-legitimate {
        background:
            linear-gradient(
                135deg,
                rgba(34, 197, 94, 0.17),
                rgba(22, 163, 74, 0.06)
            );
        border: 1px solid rgba(74, 222, 128, 0.40);
    }

    .decision-fraudulent {
        background:
            linear-gradient(
                135deg,
                rgba(239, 68, 68, 0.18),
                rgba(185, 28, 28, 0.07)
            );
        border: 1px solid rgba(248, 113, 113, 0.42);
    }

    .decision-review {
        background:
            linear-gradient(
                135deg,
                rgba(245, 158, 11, 0.18),
                rgba(180, 83, 9, 0.07)
            );
        border: 1px solid rgba(251, 191, 36, 0.42);
    }

    .decision-title {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .decision-text {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.65;
    }

    .signal-card {
        background:
            linear-gradient(
                145deg,
                rgba(15, 29, 46, 0.94),
                rgba(10, 22, 36, 0.94)
            );
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 16px;
        padding: 18px;
        min-height: 180px;
        margin-bottom: 12px;
    }

    .signal-title {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 750;
        margin-bottom: 12px;
    }

    .warning-item {
        background: rgba(239, 68, 68, 0.07);
        border-left: 3px solid #f87171;
        border-radius: 7px;
        color: #fecaca;
        padding: 9px 11px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }

    .positive-item {
        background: rgba(34, 197, 94, 0.07);
        border-left: 3px solid #4ade80;
        border-radius: 7px;
        color: #bbf7d0;
        padding: 9px 11px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }

    .empty-item {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    .model-status {
        background: rgba(34, 197, 94, 0.07);
        border: 1px solid rgba(34, 197, 94, 0.28);
        padding: 12px;
        border-radius: 12px;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .model-status-title {
        color: #4ade80;
        font-weight: 750;
        font-size: 0.9rem;
    }

    .model-status-text {
        color: #94a3b8;
        font-size: 0.77rem;
        margin-top: 4px;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.13);
    }

    ::-webkit-scrollbar {
        width: 9px;
        height: 9px;
    }

    ::-webkit-scrollbar-track {
        background: #07111f;
    }

    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. NLTK SETUP
# ============================================================

@st.cache_resource
def prepare_nltk():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(
                package_name,
                quiet=True,
            )

    stop_words = set(
        stopwords.words("english")
    )

    lemmatizer = WordNetLemmatizer()

    return stop_words, lemmatizer


try:
    STOP_WORDS, LEMMATIZER = prepare_nltk()

except Exception as error:
    st.error(
        "Text-processing resources could not be loaded."
    )
    st.exception(error)
    st.stop()


# ============================================================
# 6. LOAD SAVED FILES
# ============================================================

@st.cache_resource
def load_model_files():
    required_files = [
        MODEL_PATH,
        TFIDF_PATH,
        ENCODER_PATH,
    ]

    missing_files = [
        str(file_path)
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing project files:\n"
            + "\n".join(missing_files)
        )

    model = joblib.load(MODEL_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_PATH)
    onehot_encoder = joblib.load(ENCODER_PATH)

    return model, tfidf_vectorizer, onehot_encoder


@st.cache_data
def load_processed_dataset():
    if not PROCESSED_DATA_PATH.exists():
        return None

    return pd.read_csv(PROCESSED_DATA_PATH)


@st.cache_data
def load_model_results():
    if not MODEL_RESULTS_PATH.exists():
        return None

    return pd.read_csv(MODEL_RESULTS_PATH)


@st.cache_data
def load_misclassified_samples():
    if not MISCLASSIFIED_PATH.exists():
        return None

    return pd.read_csv(MISCLASSIFIED_PATH)


# ============================================================
# 7. GENERAL HELPERS
# ============================================================

def safe_text(value, default=""):
    if value is None or pd.isna(value):
        return default

    text = str(value).strip()

    return text if text else default


def clean_text(text):
    if text is None or pd.isna(text):
        text = ""

    text = str(text).lower()

    text = re.sub(
        r"<.*?>",
        " ",
        text,
    )

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text,
    )

    text = re.sub(
        r"\S+@\S+",
        " ",
        text,
    )

    text = re.sub(
        r"\d+",
        " ",
        text,
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation,
        )
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    words = text.split()

    cleaned_words = [
        LEMMATIZER.lemmatize(word)
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(cleaned_words)


def get_category_options(
    dataset,
    column_name,
    fallback_options,
):
    if (
        dataset is not None
        and column_name in dataset.columns
    ):
        values = (
            dataset[column_name]
            .dropna()
            .astype(str)
            .str.strip()
        )

        values = sorted(
            value
            for value in values.unique()
            if value
            and value.lower() != "nan"
        )

        if values:
            return ["Unknown"] + [
                value
                for value in values
                if value != "Unknown"
            ]

    return fallback_options


# ============================================================
# 8. FEATURE CREATION
# ============================================================

def create_model_features(
    job_data,
    tfidf_vectorizer,
    onehot_encoder,
):
    combined_text = " ".join(
        safe_text(
            job_data.get(column, "")
        )
        for column in TEXT_COLUMNS
    )

    cleaned_text = clean_text(
        combined_text
    )

    text_length = len(
        cleaned_text
    )

    words = cleaned_text.split()

    word_count = len(words)

    if word_count > 0:
        avg_word_length = float(
            np.mean(
                [
                    len(word)
                    for word in words
                ]
            )
        )
    else:
        avg_word_length = 0.0

    salary_range_missing = int(
        not safe_text(
            job_data.get(
                "salary_range",
                "",
            )
        )
    )

    department_missing = int(
        not safe_text(
            job_data.get(
                "department",
                "",
            )
        )
    )

    company_profile_missing = int(
        not safe_text(
            job_data.get(
                "company_profile",
                "",
            )
        )
    )

    requirements_missing = int(
        not safe_text(
            job_data.get(
                "requirements",
                "",
            )
        )
    )

    benefits_missing = int(
        not safe_text(
            job_data.get(
                "benefits",
                "",
            )
        )
    )

    text_features = tfidf_vectorizer.transform(
        [cleaned_text]
    )

    categorical_dataframe = pd.DataFrame(
        [
            {
                "employment_type": safe_text(
                    job_data.get(
                        "employment_type",
                        "Unknown",
                    ),
                    "Unknown",
                ),
                "required_experience": safe_text(
                    job_data.get(
                        "required_experience",
                        "Unknown",
                    ),
                    "Unknown",
                ),
                "required_education": safe_text(
                    job_data.get(
                        "required_education",
                        "Unknown",
                    ),
                    "Unknown",
                ),
                "industry": safe_text(
                    job_data.get(
                        "industry",
                        "Unknown",
                    ),
                    "Unknown",
                ),
                "function": safe_text(
                    job_data.get(
                        "function",
                        "Unknown",
                    ),
                    "Unknown",
                ),
            }
        ],
        columns=CATEGORICAL_COLUMNS,
    )

    categorical_features = onehot_encoder.transform(
        categorical_dataframe
    )

    numerical_dataframe = pd.DataFrame(
        [
            {
                "text_length": text_length,
                "word_count": word_count,
                "avg_word_length": avg_word_length,
                "telecommuting": int(
                    job_data.get(
                        "telecommuting",
                        0,
                    )
                ),
                "has_company_logo": int(
                    job_data.get(
                        "has_company_logo",
                        0,
                    )
                ),
                "has_questions": int(
                    job_data.get(
                        "has_questions",
                        0,
                    )
                ),
                "salary_range_missing":
                    salary_range_missing,
                "department_missing":
                    department_missing,
                "company_profile_missing":
                    company_profile_missing,
                "requirements_missing":
                    requirements_missing,
                "benefits_missing":
                    benefits_missing,
            }
        ],
        columns=NUMERICAL_COLUMNS,
    )

    final_features = hstack(
        [
            text_features,
            categorical_features,
            numerical_dataframe.values,
        ]
    ).tocsr()

    feature_information = {
        "cleaned_text": cleaned_text,
        "text_length": text_length,
        "word_count": word_count,
        "average_word_length":
            avg_word_length,
        "text_features":
            text_features.shape[1],
        "categorical_features":
            categorical_features.shape[1],
        "numerical_features":
            numerical_dataframe.shape[1],
        "total_features":
            final_features.shape[1],
    }

    return final_features, feature_information


# ============================================================
# 9. MODEL PREDICTION
# ============================================================

def predict_with_model(
    model,
    features,
):
    final_prediction = int(
        model.predict(features)[0]
    )

    individual_predictions = []

    if hasattr(model, "estimators_"):

        for estimator in model.estimators_:

            try:
                prediction = int(
                    estimator.predict(
                        features
                    )[0]
                )

                individual_predictions.append(
                    prediction
                )

            except Exception:
                continue

    if len(individual_predictions) >= 2:

        models_agree = (
            len(
                set(
                    individual_predictions
                )
            ) == 1
        )

        agreement = (
            individual_predictions.count(
                final_prediction
            )
            / len(
                individual_predictions
            )
        )

    else:
        models_agree = True
        agreement = 1.0

    model_decision = (
        "Potentially Fraudulent"
        if final_prediction == 1
        else "Likely Legitimate"
    )

    return {
        "prediction": final_prediction,
        "model_decision": model_decision,
        "models_agree": models_agree,
        "agreement": agreement,
        "individual_predictions":
            individual_predictions,
    }


# ============================================================
# 10. CONSISTENCY CHECKS
# ============================================================

def perform_consistency_checks(
    job_data,
):
    warnings = []
    positive_signals = []

    title = safe_text(
        job_data.get("title", "")
    )

    company_profile = safe_text(
        job_data.get(
            "company_profile",
            "",
        )
    )

    description = safe_text(
        job_data.get(
            "description",
            "",
        )
    )

    requirements = safe_text(
        job_data.get(
            "requirements",
            "",
        )
    )

    benefits = safe_text(
        job_data.get(
            "benefits",
            "",
        )
    )

    salary_range = safe_text(
        job_data.get(
            "salary_range",
            "",
        )
    )

    department = safe_text(
        job_data.get(
            "department",
            "",
        )
    )

    combined_text = " ".join(
        [
            title,
            company_profile,
            description,
            requirements,
            benefits,
        ]
    ).lower()

    if len(title) < 3:
        warnings.append(
            "Job title is missing or too short."
        )

    if len(description) < 100:
        warnings.append(
            "Job description is very short."
        )

    if not company_profile:
        warnings.append(
            "Company profile is missing."
        )

    if not requirements:
        warnings.append(
            "Job requirements are missing."
        )

    if not benefits:
        warnings.append(
            "Benefits information is missing."
        )

    if not salary_range:
        warnings.append(
            "Salary range is not specified."
        )

    if not department:
        warnings.append(
            "Department is not specified."
        )

    if int(
        job_data.get(
            "has_company_logo",
            0,
        )
    ) == 0:
        warnings.append(
            "No company logo is reported."
        )
    else:
        positive_signals.append(
            "A company logo is reported."
        )

    if int(
        job_data.get(
            "has_questions",
            0,
        )
    ) == 0:
        warnings.append(
            "No screening questions are reported."
        )
    else:
        positive_signals.append(
            "Screening questions are included."
        )

    suspicious_phrases = [
        "easy money",
        "earn money fast",
        "guaranteed income",
        "registration fee",
        "processing fee",
        "send money",
        "wire transfer",
        "western union",
        "investment required",
        "no experience needed",
        "act immediately",
        "limited positions",
        "make thousands",
        "immediate payment",
    ]

    detected_phrases = [
        phrase
        for phrase in suspicious_phrases
        if phrase in combined_text
    ]

    if detected_phrases:
        warnings.append(
            "Suspicious phrases detected: "
            + ", ".join(
                detected_phrases
            )
        )

    remote_phrases = [
        "remote",
        "work from home",
        "home based",
        "home-based",
        "telecommuting",
    ]

    text_says_remote = any(
        phrase in combined_text
        for phrase in remote_phrases
    )

    metadata_says_remote = bool(
        job_data.get(
            "telecommuting",
            0,
        )
    )

    if (
        text_says_remote
        != metadata_says_remote
    ):
        warnings.append(
            "Remote-work text does not match "
            "the telecommuting metadata."
        )

    elif (
        text_says_remote
        and metadata_says_remote
    ):
        positive_signals.append(
            "Remote-work text matches the metadata."
        )

    return warnings, positive_signals


# ============================================================
# 11. RULE-BASED RISK
# ============================================================

def calculate_rule_risk(
    job_data,
):
    risk_score = 0
    reasons = []

    title = safe_text(
        job_data.get("title", "")
    )

    company_profile = safe_text(
        job_data.get(
            "company_profile",
            "",
        )
    )

    description = safe_text(
        job_data.get(
            "description",
            "",
        )
    )

    requirements = safe_text(
        job_data.get(
            "requirements",
            "",
        )
    )

    benefits = safe_text(
        job_data.get(
            "benefits",
            "",
        )
    )

    salary_range = safe_text(
        job_data.get(
            "salary_range",
            "",
        )
    )

    department = safe_text(
        job_data.get(
            "department",
            "",
        )
    )

    combined_text = " ".join(
        [
            title,
            company_profile,
            description,
            requirements,
            benefits,
        ]
    ).lower()

    if len(title) < 3:
        risk_score += 10
        reasons.append(
            "Missing or very short job title: +10"
        )

    if len(description) < 100:
        risk_score += 15
        reasons.append(
            "Very short job description: +15"
        )

    if not company_profile:
        risk_score += 12
        reasons.append(
            "Missing company profile: +12"
        )

    if not requirements:
        risk_score += 10
        reasons.append(
            "Missing requirements: +10"
        )

    if not benefits:
        risk_score += 5
        reasons.append(
            "Missing benefits: +5"
        )

    if not salary_range:
        risk_score += 5
        reasons.append(
            "Missing salary range: +5"
        )

    if not department:
        risk_score += 4
        reasons.append(
            "Missing department: +4"
        )

    suspicious_phrase_points = {
        "easy money": 15,
        "earn money fast": 20,
        "guaranteed income": 20,
        "registration fee": 25,
        "processing fee": 25,
        "send money": 30,
        "wire transfer": 30,
        "western union": 30,
        "investment required": 25,
        "no experience needed": 10,
        "act immediately": 10,
        "limited positions": 5,
        "make thousands": 15,
        "immediate payment": 15,
    }

    for phrase, points in (
        suspicious_phrase_points.items()
    ):
        if phrase in combined_text:
            risk_score += points
            reasons.append(
                f"Suspicious phrase "
                f"'{phrase}': +{points}"
            )

    if int(
        job_data.get(
            "has_company_logo",
            0,
        )
    ) == 0:
        risk_score += 5
        reasons.append(
            "No company logo: +5"
        )

    if int(
        job_data.get(
            "has_questions",
            0,
        )
    ) == 0:
        risk_score += 4
        reasons.append(
            "No screening questions: +4"
        )

    remote_phrases = [
        "remote",
        "work from home",
        "home based",
        "home-based",
        "telecommuting",
    ]

    text_says_remote = any(
        phrase in combined_text
        for phrase in remote_phrases
    )

    metadata_says_remote = bool(
        job_data.get(
            "telecommuting",
            0,
        )
    )

    if (
        text_says_remote
        != metadata_says_remote
    ):
        risk_score += 15
        reasons.append(
            "Remote-work text and metadata mismatch: +15"
        )

    return min(risk_score, 100), reasons


# ============================================================
# 12. HYBRID DECISION
# ============================================================

def create_hybrid_decision(
    model_result,
    rule_risk_score,
):
    if rule_risk_score >= 70:
        return {
            "decision":
                "Potentially Fraudulent",
            "explanation":
                "High-risk language, missing information, "
                "or metadata inconsistencies were detected.",
        }

    if rule_risk_score >= 40:
        return {
            "decision":
                "Needs Human Review",
            "explanation":
                "Important warning signals were detected. "
                "Manual verification is recommended.",
        }

    if not model_result[
        "models_agree"
    ]:
        return {
            "decision":
                "Needs Human Review",
            "explanation":
                "The internal machine-learning models "
                "produced different predictions.",
        }

    if model_result[
        "prediction"
    ] == 1:
        return {
            "decision":
                "Potentially Fraudulent",
            "explanation":
                "The trained hybrid classifier identified "
                "the posting as potentially fraudulent.",
        }

    return {
        "decision":
            "Likely Legitimate",
        "explanation":
            "The trained classifier predicted a legitimate "
            "posting and the supporting risk level was low.",
    }


# ============================================================
# 13. CUSTOM DISPLAY FUNCTIONS
# ============================================================

def display_decision_card(
    decision,
    explanation,
):
    safe_explanation = html.escape(
        explanation
    )

    if decision == "Likely Legitimate":
        css_class = (
            "decision-legitimate"
        )
        icon = "✅"

    elif decision == "Potentially Fraudulent":
        css_class = (
            "decision-fraudulent"
        )
        icon = "🚨"

    else:
        css_class = "decision-review"
        icon = "⚠️"

    st.markdown(
        f"""
        <div class="decision-card {css_class}">
            <div class="decision-title">
                {icon} {html.escape(decision)}
            </div>
            <div class="decision-text">
                {safe_explanation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_signal_card(
    title,
    items,
    signal_type,
):
    if signal_type == "warning":
        item_class = "warning-item"
        icon = "⚠"
    else:
        item_class = "positive-item"
        icon = "✓"

    if items:
        item_html = "".join(
            f"""
            <div class="{item_class}">
                {icon} {html.escape(str(item))}
            </div>
            """
            for item in items
        )
    else:
        message = (
            "No major warning signal detected."
            if signal_type == "warning"
            else "No strong positive signal detected."
        )

        item_html = (
            f'<div class="empty-item">'
            f'{message}</div>'
        )

    st.markdown(
        f"""
        <div class="signal-card">
            <div class="signal-title">
                {html.escape(title)}
            </div>
            {item_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 14. LOAD MODEL AND DATA
# ============================================================

try:
    MODEL, TFIDF, ENCODER = (
        load_model_files()
    )

except Exception as error:
    st.error(
        "The saved model files could not be loaded."
    )

    st.code(
        str(error)
    )

    st.stop()


PROCESSED_DATASET = (
    load_processed_dataset()
)


# ============================================================
# 15. FORM OPTIONS
# ============================================================

employment_type_options = (
    get_category_options(
        PROCESSED_DATASET,
        "employment_type",
        [
            "Unknown",
            "Full-time",
            "Part-time",
            "Contract",
            "Temporary",
            "Other",
        ],
    )
)

experience_options = (
    get_category_options(
        PROCESSED_DATASET,
        "required_experience",
        [
            "Unknown",
            "Not Applicable",
            "Internship",
            "Entry level",
            "Associate",
            "Mid-Senior level",
            "Director",
            "Executive",
        ],
    )
)

education_options = (
    get_category_options(
        PROCESSED_DATASET,
        "required_education",
        [
            "Unknown",
            "Unspecified",
            "High School or equivalent",
            "Vocational",
            "Associate Degree",
            "Bachelor's Degree",
            "Master's Degree",
            "Professional",
            "Doctorate",
        ],
    )
)

industry_options = (
    get_category_options(
        PROCESSED_DATASET,
        "industry",
        [
            "Unknown",
            "Computer Software",
            "Information Technology and Services",
            "Financial Services",
            "Education Management",
        ],
    )
)

function_options = (
    get_category_options(
        PROCESSED_DATASET,
        "function",
        [
            "Unknown",
            "Information Technology",
            "Engineering",
            "Sales",
            "Marketing",
            "Administrative",
        ],
    )
)


# ============================================================
# 16. SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        padding: 10px 4px 18px 4px;
        font-size: 1.45rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.3;
    ">
        🔎 Fake Job Detection System
    </div>
    """,
    unsafe_allow_html=True,
)

selected_page = st.sidebar.radio(
    "Navigation",
    [
        "🔎 Job Prediction",
        "📊 Processed Dataset",
        "📈 Model Performance",
        "ℹ️ About the Project",
    ],
)

st.sidebar.caption(
    "A Hybrid Framework for "
    "Fake Job Posting Detection"
)


# ============================================================
# 17. JOB PREDICTION PAGE
# ============================================================

if selected_page == "🔎 Job Prediction":

    st.markdown(
        """
        <div class="section-label">
            Fraud analysis
        </div>

        <div class="main-title">
            Fake Job Detecting Dashboard
        </div>

        <div class="subtitle">
            Analyze job-posting text, structured metadata,
            missing information and consistency signals using
            a hybrid machine-learning framework.
        </div>
        """,
        unsafe_allow_html=True,
    )


    with st.form(
        "job_prediction_form"
    ):

        st.subheader(
            "Job Advertisement Text"
        )

        title = st.text_input(
            "Job title",
            placeholder=(
                "Example: Junior Software Engineer"
            ),
        )

        company_profile = st.text_area(
            "Company profile",
            placeholder=(
                "Enter information about the company."
            ),
            height=100,
        )

        description = st.text_area(
            "Job description",
            placeholder=(
                "Paste the complete job description."
            ),
            height=180,
        )

        requirements = st.text_area(
            "Requirements",
            placeholder=(
                "Enter required skills, education "
                "and experience."
            ),
            height=120,
        )

        benefits = st.text_area(
            "Benefits",
            placeholder=(
                "Enter salary, insurance, bonuses "
                "and other benefits."
            ),
            height=100,
        )

        st.subheader(
            "Structured Metadata"
        )

        left_column, right_column = (
            st.columns(2)
        )

        with left_column:

            location = st.text_input(
                "Location",
                placeholder=(
                    "Example: Dhaka, Bangladesh"
                ),
            )

            department = st.text_input(
                "Department",
                placeholder=(
                    "Example: Engineering"
                ),
            )

            salary_range = st.text_input(
                "Salary range",
                placeholder=(
                    "Example: BDT 30000-50000"
                ),
            )

            employment_type = st.selectbox(
                "Employment type",
                employment_type_options,
            )

            required_experience = st.selectbox(
                "Required experience",
                experience_options,
            )

        with right_column:

            required_education = st.selectbox(
                "Required education",
                education_options,
            )

            industry = st.selectbox(
                "Industry",
                industry_options,
            )

            job_function = st.selectbox(
                "Job function",
                function_options,
            )

            telecommuting = st.checkbox(
                "Remote or telecommuting job"
            )

            has_company_logo = st.checkbox(
                "Company logo is available"
            )

            has_questions = st.checkbox(
                "Screening questions are included"
            )

        analyze_button = (
            st.form_submit_button(
                "Analyze Job Posting",
                type="primary",
                use_container_width=True,
            )
        )

    if analyze_button:

        if (
            not title.strip()
            and not description.strip()
        ):
            st.warning(
                "Enter at least a job title "
                "or job description."
            )

        else:

            job_data = {
                "title": title,
                "company_profile":
                    company_profile,
                "description": description,
                "requirements": requirements,
                "benefits": benefits,
                "location": location,
                "department": department,
                "salary_range": salary_range,
                "employment_type":
                    employment_type,
                "required_experience":
                    required_experience,
                "required_education":
                    required_education,
                "industry": industry,
                "function": job_function,
                "telecommuting":
                    int(telecommuting),
                "has_company_logo":
                    int(has_company_logo),
                "has_questions":
                    int(has_questions),
            }

            try:

                with st.spinner(
                    "Analyzing the job posting..."
                ):

                    features, feature_info = (
                        create_model_features(
                            job_data,
                            TFIDF,
                            ENCODER,
                        )
                    )

                    expected_features = getattr(
                        MODEL,
                        "n_features_in_",
                        None,
                    )

                    if (
                        expected_features
                        is not None
                        and features.shape[1]
                        != expected_features
                    ):
                        st.error(
                            "The generated feature count "
                            "does not match the saved model."
                        )

                        st.write(
                            "Generated features:",
                            features.shape[1],
                        )

                        st.write(
                            "Expected features:",
                            expected_features,
                        )

                        st.stop()

                    model_result = (
                        predict_with_model(
                            MODEL,
                            features,
                        )
                    )

                    warnings, positive_signals = (
                        perform_consistency_checks(
                            job_data
                        )
                    )

                    rule_risk_score, risk_reasons = (
                        calculate_rule_risk(
                            job_data
                        )
                    )

                    hybrid_result = (
                        create_hybrid_decision(
                            model_result,
                            rule_risk_score,
                        )
                    )

                st.divider()

                st.subheader(
                    "Analysis Result"
                )

                final_decision = hybrid_result[
                    "decision"
                ]

                display_decision_card(
                    final_decision,
                    hybrid_result[
                        "explanation"
                    ],
                )
                result_col1, result_col2, result_col3 = st.columns(3)

                result_col1.metric(
                "Final Decision",
                final_decision,
                )

                result_col2.metric(
                "Risk Score",
                f"{rule_risk_score}/100",
                )

                result_col3.metric(
                "Model Agreement",
                f"{model_result['agreement']:.0%}",
                )

            
                if model_result[
                    "individual_predictions"
                ]:

                    with st.expander(
                        "Internal Model Predictions"
                    ):

                        for index, prediction in enumerate(
                            model_result[
                                "individual_predictions"
                            ],
                            start=1,
                        ):

                            prediction_name = (
                                "Fraudulent"
                                if prediction == 1
                                else "Legitimate"
                            )

                            st.write(
                                f"Model {index}: "
                                f"{prediction_name}"
                            )

                with st.expander(
                    "Risk Score Breakdown"
                ):

                    st.write(
                        f"Total risk score: "
                        f"{rule_risk_score}/100"
                    )

                    if risk_reasons:

                        for reason in risk_reasons:
                            st.write(
                                f"• {reason}"
                            )

                    else:
                        st.write(
                            "No rule-based risk factors "
                            "were detected."
                        )

                st.subheader(
                    "Consistency Analysis"
                )

                warning_column, positive_column = (
                    st.columns(2)
                )

                with warning_column:
                    display_signal_card(
                        "Warning Signals",
                        warnings,
                        "warning",
                    )

                with positive_column:
                    display_signal_card(
                        "Positive Signals",
                        positive_signals,
                        "positive",
                    )

                with st.expander(
                    "Generated Feature Information"
                ):

                    feature1, feature2, feature3 = (
                        st.columns(3)
                    )

                    feature1.metric(
                        "TF-IDF features",
                        feature_info[
                            "text_features"
                        ],
                    )

                    feature2.metric(
                        "Metadata features",
                        feature_info[
                            "categorical_features"
                        ],
                    )

                    feature3.metric(
                        "Numerical features",
                        feature_info[
                            "numerical_features"
                        ],
                    )

                    st.write(
                        "Total generated features:",
                        feature_info[
                            "total_features"
                        ],
                    )

                    st.write(
                        "Cleaned word count:",
                        feature_info[
                            "word_count"
                        ],
                    )

                with st.expander(
                    "Cleaned Text"
                ):

                    cleaned_text = feature_info[
                        "cleaned_text"
                    ]

                    if cleaned_text:
                        st.write(
                            cleaned_text
                        )
                    else:
                        st.write(
                            "No cleaned text was generated."
                        )

            except Exception as error:

                st.error(
                    "Prediction could not be completed."
                )

                st.exception(error)


# ============================================================
# 18. PROCESSED DATASET PAGE
# ============================================================

elif selected_page == "📊 Processed Dataset":

    st.markdown(
        """
        <div class="section-label">
            Data intelligence
        </div>

        <div class="main-title">
            Processed Dataset Overview
        </div>

        <div class="subtitle">
            Explore the cleaned and transformed job-posting
            records used by the project.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if PROCESSED_DATASET is None:

        st.error(
            "The processed dataset could not be found."
        )

    else:

        total_records = len(
            PROCESSED_DATASET
        )

        total_columns = len(
            PROCESSED_DATASET.columns
        )

        fraudulent_count = None
        legitimate_count = None

        if (
            "fraudulent"
            in PROCESSED_DATASET.columns
        ):

            target_values = pd.to_numeric(
                PROCESSED_DATASET[
                    "fraudulent"
                ],
                errors="coerce",
            ).fillna(0)

            fraudulent_count = int(
                target_values.sum()
            )

            legitimate_count = (
                total_records
                - fraudulent_count
            )

        metric1, metric2, metric3, metric4 = (
            st.columns(4)
        )

        metric1.metric(
            "Processed records",
            f"{total_records:,}",
        )

        metric2.metric(
            "Dataset columns",
            f"{total_columns:,}",
        )

        metric3.metric(
            "Legitimate jobs",
            (
                f"{legitimate_count:,}"
                if legitimate_count
                is not None
                else "N/A"
            ),
        )

        metric4.metric(
            "Fraudulent jobs",
            (
                f"{fraudulent_count:,}"
                if fraudulent_count
                is not None
                else "N/A"
            ),
        )

        st.divider()

        if (
            "fraudulent"
            in PROCESSED_DATASET.columns
        ):

            st.subheader(
                "Class Distribution"
            )

            class_distribution = (
                pd.to_numeric(
                    PROCESSED_DATASET[
                        "fraudulent"
                    ],
                    errors="coerce",
                )
                .map(
                    {
                        0: "Legitimate",
                        1: "Fraudulent",
                    }
                )
                .value_counts()
            )

            st.bar_chart(
                class_distribution
            )

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            PROCESSED_DATASET.head(20),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "Dataset Columns"
        ):
            st.write(
                list(
                    PROCESSED_DATASET.columns
                )
            )

        st.subheader(
            "Missing Values"
        )

        missing_values = (
            PROCESSED_DATASET
            .isnull()
            .sum()
            .reset_index()
        )

        missing_values.columns = [
            "Column",
            "Missing values",
        ]

        missing_values = (
            missing_values
            .sort_values(
                "Missing values",
                ascending=False,
            )
        )

        st.dataframe(
            missing_values,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 19. MODEL PERFORMANCE PAGE
# ============================================================

elif selected_page == "📈 Model Performance":

    st.markdown(
        """
        <div class="section-label">
            Evaluation analytics
        </div>

        <div class="main-title">
            Model Performance
        </div>

        <div class="subtitle">
            Compare the performance of the trained
            classification models used in the project.
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_results = (
        load_model_results()
    )

    if model_results is None:

        st.error(
            "The model comparison file "
            "could not be found."
        )

    else:

        st.subheader(
            "Model Comparison"
        )

        st.dataframe(
            model_results,
            use_container_width=True,
            hide_index=True,
        )

        if (
            "Model" in model_results.columns
            and "F1-Score" in model_results.columns
        ):

            st.subheader(
                "F1-Score Comparison"
            )

            f1_data = (
                model_results[
                    [
                        "Model",
                        "F1-Score",
                    ]
                ]
                .set_index(
                    "Model"
                )
            )

            st.bar_chart(
                f1_data
            )

        if (
            "Model" in model_results.columns
            and "Precision"
            in model_results.columns
            and "Recall"
            in model_results.columns
        ):

            st.subheader(
                "Precision and Recall"
            )

            precision_recall_data = (
                model_results[
                    [
                        "Model",
                        "Precision",
                        "Recall",
                    ]
                ]
                .set_index(
                    "Model"
                )
            )

            st.bar_chart(
                precision_recall_data
            )

    misclassified_samples = (
        load_misclassified_samples()
    )

    if misclassified_samples is not None:

        st.divider()

        st.subheader(
            "Misclassified Samples"
        )

        st.metric(
            "Saved error samples",
            f"{len(misclassified_samples):,}",
        )

        st.dataframe(
            misclassified_samples.head(30),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 20. ABOUT PAGE
# ============================================================

elif selected_page == "ℹ️ About the Project":

    st.markdown(
        """
        <div class="section-label">
            System documentation
        </div>

        <div class="main-title">
            About the Project
        </div>

        <div class="subtitle">
            A hybrid data-mining framework for detecting
            fraudulent online job advertisements.
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview1, overview2, overview3 = (
        st.columns(3)
    )

    overview1.metric(
        "Framework",
        "Hybrid",
    )

    overview2.metric(
        "Input sources",
        "Multiple",
    )

    overview3.metric(
        "Decision classes",
        "3",
    )

    st.divider()

    st.subheader(
        "System Workflow"
    )

    st.code(
        """
Job-posting information
           |
           v
Text cleaning and lemmatization
           |
           v
TF-IDF feature extraction
           |
           v
Structured metadata encoding
           |
           v
Numerical and missing-value features
           |
           v
Hybrid machine-learning classifier
           |
           v
Risk and consistency analysis
           |
           v
Legitimate / Fraudulent / Human Review
        """,
        language="text",
    )

    st.subheader(
        "Information Sources"
    )

    source1, source2, source3 = (
        st.columns(3)
    )

    with source1:
        st.markdown(
            """
            ### Text Information

            - Job title
            - Company profile
            - Job description
            - Requirements
            - Benefits
            """
        )

    with source2:
        st.markdown(
            """
            ### Structured Metadata

            - Employment type
            - Experience level
            - Education level
            - Industry
            - Job function
            - Company logo
            """
        )

    with source3:
        st.markdown(
            """
            ### Consistency Signals

            - Missing information
            - Suspicious language
            - Remote-work mismatch
            - Screening questions
            - Internal model agreement
            """
        )

    st.subheader(
        "Decision Strategy"
    )

    st.write(
        """
        The system combines the trained classifier result,
        agreement between internal models, suspicious-language
        checks, missing-information indicators and metadata
        consistency signals.
        """
    )

    st.write(
        """
        Cases with conflicting or moderately risky evidence
        are marked for human review instead of being treated
        as certain predictions.
        """
    )