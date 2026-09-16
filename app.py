import os
import time
import json
import re
from html import unescape
from html.parser import HTMLParser
from typing import Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from streamlit_lottie import st_lottie

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from gemini_helper import generate_cardio_response


# =========================================================
# CONSTANTS & CONFIGURATION
# =========================================================

DATA_PATH = "train.csv"
TEST_PATH = "test.csv"

MODEL_PATH = "cardiovascular_rf_model.pkl"
THRESHOLD_PATH = "threshold.pkl"

FEATURE_COLUMNS = [
    "age",
    "education",
    "sex",
    "is_smoking",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

CATEGORICAL_FEATURES = [
    "education",
    "sex",
    "is_smoking",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
]

NUMERIC_FEATURES = [
    "age",
    "cigsPerDay",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

PLOT_COLOR_PRIMARY = "#2563eb"
PLOT_COLOR_SECONDARY = "#dc2626"
PLOT_TEMPLATE = "plotly_white"

GEMINI_API_KEY_NAME = "GEMINI_API_KEY"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Cardiovascular Risk AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# SESSION STATE
# =========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "patient_context" not in st.session_state:
    st.session_state.patient_context = None

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_probability" not in st.session_state:
    st.session_state.last_probability = None

if "last_is_high_risk" not in st.session_state:
    st.session_state.last_is_high_risk = None

if "last_patient_data" not in st.session_state:
    st.session_state.last_patient_data = None


# =========================================================
# HTML / AI RENDERING HELPERS
# =========================================================

def render_html(html_content: str) -> None:
    """
    Render HTML correctly inside Streamlit.
    Uses st.html() when available and falls back
    to st.markdown(..., unsafe_allow_html=True).
    """

    if not html_content:
        return

    try:
        if hasattr(st, "html"):
            st.html(html_content)
        else:
            st.markdown(
                html_content,
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            html_content,
            unsafe_allow_html=True,
        )


class SimpleHTMLToMarkdown(HTMLParser):
    """
    Convert common HTML elements from AI responses
    into Markdown so HTML tags never appear as raw text.
    """

    def __init__(self):
        super().__init__()
        self.output = []
        self.list_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()

        if tag in ("h1", "h2", "h3"):
            level = int(tag[1])
            self.output.append("\n" + ("#" * level) + " ")

        elif tag == "h4":
            self.output.append("\n#### ")

        elif tag in ("p", "div", "section"):
            self.output.append("\n")

        elif tag == "br":
            self.output.append("\n")

        elif tag == "strong" or tag == "b":
            self.output.append("**")

        elif tag == "em" or tag == "i":
            self.output.append("*")

        elif tag == "ul":
            self.list_depth += 1
            self.output.append("\n")

        elif tag == "ol":
            self.list_depth += 1
            self.output.append("\n")

        elif tag == "li":
            self.output.append("\n- ")

        elif tag == "code":
            self.output.append("`")

        elif tag == "pre":
            self.output.append("\n```text\n")

    def handle_endtag(self, tag):
        tag = tag.lower()

        if tag in ("h1", "h2", "h3", "h4"):
            self.output.append("\n")

        elif tag in ("p", "div", "section"):
            self.output.append("\n")

        elif tag == "strong" or tag == "b":
            self.output.append("**")

        elif tag == "em" or tag == "i":
            self.output.append("*")

        elif tag in ("ul", "ol"):
            self.list_depth = max(0, self.list_depth - 1)
            self.output.append("\n")

        elif tag == "code":
            self.output.append("`")

        elif tag == "pre":
            self.output.append("\n```\n")

    def handle_data(self, data):
        self.output.append(data)

    def get_markdown(self):
        text = "".join(self.output)

        text = unescape(text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()


def normalize_ai_response(content: Any) -> str:
    """
    Normalize Gemini output before displaying it.

    If Gemini returns HTML, convert common HTML tags
    to Markdown. If it returns Markdown normally,
    keep it unchanged.
    """

    if content is None:
        return ""

    text = str(content).strip()

    if not text:
        return ""

    html_pattern = re.compile(
        r"<\s*/?\s*(html|body|div|p|h1|h2|h3|h4|"
        r"strong|b|em|i|ul|ol|li|br|code|pre|span)"
        r"[^>]*>",
        re.IGNORECASE,
    )

    if html_pattern.search(text):

        parser = SimpleHTMLToMarkdown()

        try:
            parser.feed(text)
            parser.close()

            converted = parser.get_markdown()

            if converted:
                return converted

        except Exception:
            pass

    # Remove HTML comments if any remain
    text = re.sub(
        r"<!--.*?-->",
        "",
        text,
        flags=re.DOTALL,
    )

    # Remove any remaining HTML tags safely
    text = re.sub(
        r"<[^>]+>",
        "",
        text,
    )

    return unescape(text).strip()


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def reset_cardio_chat() -> None:
    """Reset CardioAI conversation history."""
    st.session_state.chat_history = []


def build_patient_context(
    age: int,
    sex: str,
    education: int,
    is_smoking: str,
    cigs_per_day: int,
    diabetes: str,
    bp_meds: str,
    stroke: str,
    hypertension: str,
    tot_chol: float,
    sys_bp: float,
    dia_bp: float,
    bmi: float,
    heart_rate: float,
    glucose: float,
    probability: float,
    threshold_value: float,
) -> str:

    return f"""
Current patient information:

Age: {age} years
Sex: {sex}
Education Level: {education}
Current Smoker: {is_smoking}
Cigarettes per Day: {cigs_per_day}
Diabetes: {diabetes}
Anti-Hypertensive Medication: {bp_meds}
Prior Stroke: {stroke}
Prevalent Hypertension: {hypertension}

Total Cholesterol: {tot_chol:.1f} mg/dL
Systolic Blood Pressure: {sys_bp:.1f} mmHg
Diastolic Blood Pressure: {dia_bp:.1f} mmHg
BMI: {bmi:.1f}
Heart Rate: {heart_rate:.1f} bpm
Glucose: {glucose:.1f} mg/dL

Machine Learning Predicted 10-Year CHD Risk:
{probability * 100:.1f}%

Configured ML Decision Threshold:
{threshold_value * 100:.1f}%

Important:
The machine learning model has already calculated the risk.
Do not recalculate, alter, or reinterpret the numerical prediction.
Use the prediction only for explanation and educational context.
"""


def render_risk_result(
    probability: float,
    is_high_risk: bool,
    threshold_value: float,
    animation_warning,
    animation_success,
) -> None:

    st.subheader("Diagnostic Results")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:

        st.metric(
            "Estimated 10-Year Risk",
            f"{probability * 100:.1f}%",
        )

        st.caption(
            f"Configured Decision Threshold: "
            f"**{threshold_value * 100:.0f}%**"
        )

        st.write("")

        if is_high_risk and animation_warning:

            st_lottie(
                animation_warning,
                height=140,
                key="warning_result_animation",
            )

        elif not is_high_risk and animation_success:

            st_lottie(
                animation_success,
                height=140,
                key="success_result_animation",
            )

    with res_col2:

        st.progress(
            min(max(probability, 0.0), 1.0)
        )

        if is_high_risk:

            render_html(
                f"""
                <div class="result-card-high">

                    <h3>
                        ⚠️ Elevated Cardiovascular Risk Detected
                    </h3>

                    <p>
                        The machine learning model estimated a
                        <strong>
                            {probability * 100:.1f}%
                        </strong>
                        10-year cardiovascular risk probability,
                        which is above the configured decision threshold.
                    </p>

                    <p>
                        This result is an AI-generated estimate for
                        academic and educational purposes and should not
                        be interpreted as a medical diagnosis.
                    </p>

                </div>
                """
            )

        else:

            render_html(
                f"""
                <div class="result-card-low">

                    <h3>
                        ✅ Lower Calculated Cardiovascular Risk
                    </h3>

                    <p>
                        The machine learning model estimated a
                        <strong>
                            {probability * 100:.1f}%
                        </strong>
                        10-year cardiovascular risk probability,
                        which is below the configured decision threshold.
                    </p>

                    <p>
                        Continue maintaining healthy habits and routine
                        healthcare follow-up.
                    </p>

                </div>
                """
            )


def render_cardio_ai() -> None:

    st.divider()

    st.subheader("🤖 CardioAI Assistant")

    st.caption(
        "Ask questions about cardiovascular health, risk factors, "
        "or the machine learning prediction."
    )

    api_key_available = bool(
        os.getenv(GEMINI_API_KEY_NAME)
    )

    if api_key_available:

        st.success(
            "CardioAI is connected and ready.",
            icon="✅",
        )

    else:

        st.error(
            "Gemini API key not found. "
            "Set GEMINI_API_KEY before using CardioAI.",
            icon="❌",
        )

    # -----------------------------------------------------
    # CURRENT PREDICTION CONTEXT
    # -----------------------------------------------------

    if st.session_state.last_probability is not None:

        probability = (
            st.session_state.last_probability
        )

        render_html(
            f"""
            <div class="ai-context-card">

                <div class="ai-context-title">
                    🧠 Current ML Context
                </div>

                <div class="ai-context-row">
                    <span>Current Estimated Risk</span>
                    <strong>
                        {probability * 100:.1f}%
                    </strong>
                </div>

                <div class="ai-context-row">
                    <span>Context Available</span>
                    <strong>Yes</strong>
                </div>

            </div>
            """
        )

    else:

        st.info(
            "No prediction has been generated yet. "
            "You can still ask CardioAI general cardiovascular questions."
        )

    # -----------------------------------------------------
    # SUGGESTED QUESTIONS
    # -----------------------------------------------------

    st.markdown("### Suggested Questions")

    q1, q2, q3 = st.columns(3)

    with q1:

        question_1 = st.button(
            "❤️ What is cardiovascular disease?",
            use_container_width=True,
            key="question_cvd",
        )

    with q2:

        question_2 = st.button(
            "📊 What does my prediction mean?",
            use_container_width=True,
            key="question_prediction",
        )

    with q3:

        question_3 = st.button(
            "🥗 How can I improve my heart health?",
            use_container_width=True,
            key="question_health",
        )

    selected_question: Optional[str] = None

    if question_1:

        selected_question = (
            "What is cardiovascular disease?"
        )

    elif question_2:

        if st.session_state.last_probability is not None:

            selected_question = (
                "What does my machine learning "
                "cardiovascular risk prediction mean?"
            )

        else:

            selected_question = (
                "Can you explain how cardiovascular "
                "risk predictions work?"
            )

    elif question_3:

        selected_question = (
            "How can I improve my cardiovascular health "
            "through healthy lifestyle habits?"
        )

    # -----------------------------------------------------
    # CHAT HISTORY
    # -----------------------------------------------------

    for message in st.session_state.chat_history:

        role = message.get("role", "")
        content = normalize_ai_response(
            message.get("content", "")
        )

        if not content:
            continue

        if role == "user":

            with st.chat_message("user"):
                st.markdown(content)

        elif role == "assistant":

            with st.chat_message("assistant"):
                st.markdown(content)

    # -----------------------------------------------------
    # USER CHAT INPUT
    # -----------------------------------------------------

    user_question = st.chat_input(
        "Ask CardioAI a question...",
        key="cardio_ai_input",
    )

    if user_question:

        selected_question = user_question

    # -----------------------------------------------------
    # GENERATE GEMINI RESPONSE
    # -----------------------------------------------------

    if selected_question:

        selected_question = (
            str(selected_question).strip()
        )

        if not selected_question:
            return

        if not api_key_available:

            st.error(
                "CardioAI cannot connect because "
                "GEMINI_API_KEY is not configured."
            )

            return

        previous_history = (
            st.session_state.chat_history.copy()
        )

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": selected_question,
            }
        )

        try:

            with st.spinner(
                "CardioAI is thinking..."
            ):

                response = generate_cardio_response(
                    user_message=selected_question,
                    chat_history=previous_history,
                    patient_context=(
                        st.session_state.patient_context
                        if st.session_state.patient_context
                        else
                        "No current patient prediction is available."
                    ),
                )

            clean_response = normalize_ai_response(
                response
            )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": clean_response,
                }
            )

            st.rerun()

        except Exception as e:

            if st.session_state.chat_history:
                st.session_state.chat_history.pop()

            st.error(
                "CardioAI could not generate a response right now."
            )

            st.caption(
                f"Technical details: {str(e)}"
            )

    # -----------------------------------------------------
    # RESET CHAT
    # -----------------------------------------------------

    st.markdown("")

    reset_col1, reset_col2 = st.columns([4, 1])

    with reset_col2:

        if st.button(
            "🔄 Reset Chat",
            use_container_width=True,
            key="reset_cardio_chat",
        ):

            reset_cardio_chat()
            st.rerun()


# =========================================================
# LOTTIE ANIMATIONS LOADER
# =========================================================

@st.cache_data
def load_lottie_file(filepath: str):

    try:

        with open(
            filepath,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception:

        return None


@st.cache_data
def load_lottie_url(url: str):

    try:

        response = requests.get(
            url,
            timeout=5,
        )

        if response.status_code == 200:
            return response.json()

    except Exception:

        return None

    return None


lottie_heart = load_lottie_file(
    "Heartbeat Lottie Animation.json"
)

lottie_success = load_lottie_url(
    "https://assets2.lottiefiles.com/packages/lf20_pqn4625x.json"
)

lottie_warning = load_lottie_url(
    "https://assets10.lottiefiles.com/packages/lf20_Tkw43i.json"
)


# =========================================================
# CUSTOM CSS & THEMING
# =========================================================

render_html(
    """
    <style>

    :root {
        --bg-main: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #0f172a;
        --text-muted: #64748b;
        --primary-blue: #1d4ed8;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero-container {
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        background:
            linear-gradient(
                135deg,
                #0f172a 0%,
                #1e3a8a 50%,
                #2563eb 100%
            );
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow:
            0 10px 25px -5px
            rgba(15, 23, 42, 0.15);
    }

    .hero-container h1 {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        color: #ffffff !important;
    }

    .hero-container p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-bottom: 0;
        max-width: 750px;
    }

    .card {
        background-color: var(--card-bg);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow:
            0 1px 3px 0
            rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }

    .result-card-high {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 6px solid #dc2626;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .result-card-low {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #16a34a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .ai-context-card {
        background:
            linear-gradient(
                135deg,
                #eff6ff 0%,
                #ffffff 100%
            );
        border: 1px solid #bfdbfe;
        border-left: 5px solid #2563eb;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .ai-context-title {
        font-size: 1rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.75rem;
    }

    .ai-context-row {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.35rem 0;
        color: #475569;
    }

    .ai-context-row strong {
        color: #0f172a;
    }

    .metric-badge {
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
    }

    div[data-testid="stSidebarHeader"] {
        padding-top: 1rem;
    }

    div[data-testid="stMetric"] {
        background-color: var(--card-bg);
        border: 1px solid #e2e8f0;
        padding: 18px;
        border-radius: 12px;
        box-shadow:
            0 4px 15px
            rgba(0, 0, 0, 0.05);
        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow:
            0 6px 20px
            rgba(37, 99, 235, 0.15);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-weight: 600;
        font-size: 0.95rem;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 800;
    }

    </style>
    """
)


# =========================================================
# DATA & MODEL LOADERS
# =========================================================

@st.cache_data(show_spinner="Loading datasets...")
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    if os.path.exists(DATA_PATH):

        try:

            df_train = pd.read_csv(
                DATA_PATH
            )

        except Exception as e:

            st.error(
                f"Failed to load `{DATA_PATH}`: {e}"
            )

    else:

        st.error(
            f"Training data file `{DATA_PATH}` not found."
        )

    if os.path.exists(TEST_PATH):

        try:

            df_test = pd.read_csv(
                TEST_PATH
            )

        except Exception as e:

            st.error(
                f"Failed to load `{TEST_PATH}`: {e}"
            )

    return df_train, df_test


@st.cache_resource(
    show_spinner="Loading machine learning model..."
)
def load_artifacts() -> Tuple[Any, float]:

    model = None
    threshold = 0.50

    if os.path.exists(MODEL_PATH):

        try:

            model = joblib.load(
                MODEL_PATH
            )

        except Exception as e:

            st.error(
                f"Failed to load model file: {e}"
            )

    else:

        st.error(
            f"Model file `{MODEL_PATH}` not found."
        )

    if os.path.exists(THRESHOLD_PATH):

        try:

            threshold = float(
                joblib.load(
                    THRESHOLD_PATH
                )
            )

        except Exception:

            threshold = 0.50

    return model, threshold


df, test_df = load_data()
model, threshold = load_artifacts()


# =========================================================
# SIDEBAR & BRANDING
# =========================================================

with st.sidebar:

    st.markdown(
        "### 🩺 Clinical AI Platform"
    )

    st.caption(
        "Cardiovascular Assessment Suite"
    )

    st.divider()

    page = st.radio(
        "Navigation Module",
        [
            "🏠 Overview",
            "📊 Exploratory Data Analysis",
            "🤖 Model Intelligence",
            "🩺 Risk Assessment Engine",
            "ℹ️ Documentation",
        ],
        index=0,
    )

    st.divider()

 

    st.markdown(
        """
        **System Status**: `Operational`  
        **Model**: Random Forest Classifier  
        **Primary Target**: 10-Yr CHD Risk
        """
    )

    if os.getenv(
        GEMINI_API_KEY_NAME
    ):

        st.success(
            "CardioAI: Connected",
            icon="🤖",
        )

    else:

        st.warning(
            "CardioAI: API Key Missing",
            icon="⚠️",
        )

    st.caption(
        "Version 3.0.0 | Academic Release"
    )


# =========================================================
# MODULE 1: OVERVIEW
# =========================================================

if page == "🏠 Overview":

    col_hero, col_anim = st.columns(
        [3, 1]
    )

    with col_hero:

        render_html(
            """
            <div class="hero-container">

                <h1>
                    Cardiovascular Risk Intelligence
                </h1>

                <p>
                    An interactive academic machine learning
                    platform designed to estimate 10-year
                    Coronary Heart Disease (CHD) risk using
                    demographic, behavioral, and clinical
                    biomarkers.
                </p>

            </div>
            """
        )

    with col_anim:

        if lottie_heart:

            st_lottie(
                lottie_heart,
                height=180,
                key="heart_pulse",
            )

    st.subheader(
        "Key Data Indicators"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Cohort Records",
            (
                f"{len(df):,}"
                if not df.empty
                else "N/A"
            ),
        )

    with col2:

        st.metric(
            "Clinical Features",
            f"{len(FEATURE_COLUMNS)}",
        )

    with col3:

        missing_count = (
            df.isnull().sum().sum()
            if not df.empty
            else 0
        )

        st.metric(
            "Missing Biomarkers",
            f"{missing_count:,}",
        )

    with col4:

        dup_count = (
            df.duplicated().sum()
            if not df.empty
            else 0
        )

        st.metric(
            "Duplicate Entries",
            f"{dup_count:,}",
        )

    st.divider()

    col_left, col_right = st.columns(
        [1.2, 1]
    )

    with col_left:

        st.subheader(
            "Pipeline Architecture"
        )

        st.markdown(
            """
            1. **Data Preprocessing & Imputation**  
               Cleaning missing clinical records.

            2. **Exploratory Biomarker Analysis**  
               Understanding distributions and relationships.

            3. **Hyperparameter Optimization**  
               Tuning the Random Forest Classifier using Grid Search.

            4. **Threshold Selection**  
               Applying the configured probability threshold.

            5. **Interactive Risk Scoring**  
               Estimating model-based cardiovascular risk.

            6. **CardioAI Assistant**  
               Explaining predictions and answering general
               cardiovascular health questions.
            """
        )

    with col_right:

        st.subheader(
            "Target Label Balance"
        )

        if (
            not df.empty
            and "TenYearCHD" in df.columns
        ):

            target_counts = (
                df["TenYearCHD"]
                .value_counts()
                .reset_index()
            )

            target_counts.columns = [
                "TenYearCHD",
                "Count",
            ]

            target_counts["Status"] = (
                target_counts["TenYearCHD"]
                .map(
                    {
                        0: "No CHD",
                        1: "CHD Risk",
                    }
                )
            )

            fig = px.pie(
                target_counts,
                names="Status",
                values="Count",
                color="Status",
                color_discrete_map={
                    "No CHD": "#3b82f6",
                    "CHD Risk": "#ef4444",
                },
                hole=0.4,
                template=PLOT_TEMPLATE,
            )

            fig.update_traces(
                textinfo="percent+label"
            )

            fig.update_layout(
                showlegend=False,
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20,
                ),
                height=280,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.info(
                "TenYearCHD target column is not available."
            )


# =========================================================
# MODULE 2: EDA
# =========================================================

elif page == "📊 Exploratory Data Analysis":

    st.title(
        "📊 Exploratory Data Analysis"
    )

    st.caption(
        "Analyze statistical distributions and "
        "biomarker relationships across patient cohorts."
    )

    if df.empty:

        st.warning(
            "No dataset loaded. Please verify "
            "`train.csv` exists."
        )

        st.stop()

    tab1, tab2, tab3 = st.tabs(
        [
            "Biomarker Explorer",
            "Correlation Matrix",
            "Cohort Summary",
        ]
    )

    with tab1:

        st.subheader(
            "Single Feature Distribution"
        )

        selected_feature = st.selectbox(
            "Select Biomarker for Evaluation",
            FEATURE_COLUMNS,
            index=0,
        )

        col1, col2 = st.columns(2)

        if selected_feature in CATEGORICAL_FEATURES:

            with col1:

                counts = (
                    df[selected_feature]
                    .value_counts()
                    .reset_index()
                )

                counts.columns = [
                    selected_feature,
                    "Patients",
                ]

                fig = px.bar(
                    counts,
                    x=selected_feature,
                    y="Patients",
                    title=(
                        f"Distribution of "
                        f"{selected_feature}"
                    ),
                    color_discrete_sequence=[
                        PLOT_COLOR_PRIMARY
                    ],
                    template=PLOT_TEMPLATE,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            with col2:

                grouped = (
                    df.groupby(
                        [
                            selected_feature,
                            "TenYearCHD",
                        ]
                    )
                    .size()
                    .reset_index(
                        name="Patients"
                    )
                )

                grouped["CHD"] = (
                    grouped["TenYearCHD"]
                    .map(
                        {
                            0: "No",
                            1: "Yes",
                        }
                    )
                )

                fig = px.bar(
                    grouped,
                    x=selected_feature,
                    y="Patients",
                    color="CHD",
                    barmode="group",
                    title=(
                        f"{selected_feature} "
                        "vs. 10-Yr CHD Outcome"
                    ),
                    color_discrete_map={
                        "No": "#3b82f6",
                        "Yes": "#ef4444",
                    },
                    template=PLOT_TEMPLATE,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        else:

            with col1:

                fig = px.histogram(
                    df,
                    x=selected_feature,
                    nbins=30,
                    title=(
                        f"{selected_feature} "
                        "Population Spread"
                    ),
                    color_discrete_sequence=[
                        PLOT_COLOR_PRIMARY
                    ],
                    template=PLOT_TEMPLATE,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            with col2:

                if "TenYearCHD" in df.columns:

                    df_temp = df.copy()

                    df_temp["CHD"] = (
                        df_temp["TenYearCHD"]
                        .map(
                            {
                                0: "No Risk",
                                1: "CHD Risk",
                            }
                        )
                    )

                    fig = px.box(
                        df_temp,
                        x="CHD",
                        y=selected_feature,
                        color="CHD",
                        title=(
                            f"{selected_feature} Variance "
                            "across Outcome Groups"
                        ),
                        color_discrete_map={
                            "No Risk": "#3b82f6",
                            "CHD Risk": "#ef4444",
                        },
                        template=PLOT_TEMPLATE,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                else:

                    st.info(
                        "TenYearCHD is not available for comparison."
                    )

    with tab2:

        st.subheader(
            "Feature Correlation Heatmap"
        )

        numeric_df = (
            df.select_dtypes(
                include=["number"]
            )
        )

        corr = numeric_df.corr()

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Blues",
            title="Pearson Correlation Matrix",
            template=PLOT_TEMPLATE,
        )

        fig_corr.update_layout(
            height=650
        )

        st.plotly_chart(
            fig_corr,
            use_container_width=True,
        )

    with tab3:

        st.subheader(
            "Raw Data Preview"
        )

        st.dataframe(
            df,
            use_container_width=True,
            height=400,
        )


# =========================================================
# MODULE 3: MODEL INTELLIGENCE
# =========================================================

elif page == "🤖 Model Intelligence":

    st.title(
        "🤖 Model Performance & Validation"
    )

    st.caption(
        "Performance evaluation, threshold calibration, "
        "feature importance, and benchmark comparison."
    )

    has_test_target = (
        not test_df.empty
        and "TenYearCHD" in test_df.columns
    )

    if model is not None and has_test_target:

        X_test = test_df[
            FEATURE_COLUMNS
        ]

        y_test = test_df[
            "TenYearCHD"
        ]

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        predictions = (
            probabilities >= threshold
        ).astype(int)

        acc = accuracy_score(
            y_test,
            predictions,
        )

        prec = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        rec = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        auc = roc_auc_score(
            y_test,
            probabilities,
        )

        m1, m2, m3, m4, m5 = st.columns(5)

        m1.metric(
            "Accuracy",
            f"{acc:.3f}",
        )

        m2.metric(
            "Precision",
            f"{prec:.3f}",
        )

        m3.metric(
            "Recall",
            f"{rec:.3f}",
        )

        m4.metric(
            "F1 Score",
            f"{f1:.3f}",
        )

        m5.metric(
            "ROC-AUC",
            f"{auc:.3f}",
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Confusion Matrix"
            )

            cm = confusion_matrix(
                y_test,
                predictions,
            )

            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=[
                    "Predicted: No",
                    "Predicted: Yes",
                ],
                y=[
                    "Actual: No",
                    "Actual: Yes",
                ],
                color_continuous_scale="Blues",
                template=PLOT_TEMPLATE,
            )

            st.plotly_chart(
                fig_cm,
                use_container_width=True,
            )

        with col2:

            st.subheader(
                "ROC Curve"
            )

            fpr, tpr, _ = roc_curve(
                y_test,
                probabilities,
            )

            fig_roc = go.Figure()

            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=(
                        "Random Forest "
                        f"(AUC = {auc:.3f})"
                    ),
                    line=dict(
                        color=PLOT_COLOR_PRIMARY,
                        width=2,
                    ),
                )
            )

            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Baseline",
                    line=dict(
                        dash="dash",
                        color="#94a3b8",
                    ),
                )
            )

            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                template=PLOT_TEMPLATE,
                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=20,
                ),
            )

            st.plotly_chart(
                fig_roc,
                use_container_width=True,
            )

    else:

        st.info(
            "Evaluation metrics are unavailable because "
            "the test dataset does not contain TenYearCHD "
            "or the model was not loaded."
        )

    st.divider()

    st.subheader(
        "Feature Importance Analysis"
    )

    if (
        model is not None
        and hasattr(
            model,
            "feature_importances_",
        )
    ):

        importances = model.feature_importances_

        if len(importances) == len(
            FEATURE_COLUMNS
        ):

            importance_df = pd.DataFrame(
                {
                    "Biomarker": FEATURE_COLUMNS,
                    "Importance": importances,
                }
            ).sort_values(
                "Importance",
                ascending=True,
            )

            fig_imp = px.bar(
                importance_df,
                x="Importance",
                y="Biomarker",
                orientation="h",
                color_discrete_sequence=[
                    PLOT_COLOR_PRIMARY
                ],
                template=PLOT_TEMPLATE,
            )

            fig_imp.update_layout(
                height=450
            )

            st.plotly_chart(
                fig_imp,
                use_container_width=True,
            )

        else:

            st.info(
                "Feature importance length does not match "
                "the configured feature list."
            )

    else:

        st.info(
            "Feature importance is not available "
            "for the loaded model."
        )

    st.divider()

    st.subheader(
        "Algorithm Benchmarking"
    )

    benchmark_data = pd.DataFrame(
        {
            "Algorithm Model": [
                "Random Forest",
                "K-Nearest Neighbors",
                "Decision Tree",
                "Logistic Regression",
                "Support Vector Machine",
            ],
            "Accuracy": [
                0.886,
                0.901,
                0.839,
                0.684,
                0.664,
            ],
            "Precision": [
                0.906,
                0.861,
                0.837,
                0.696,
                0.661,
            ],
            "Recall": [
                0.871,
                0.965,
                0.858,
                0.696,
                0.728,
            ],
            "F1 Score": [
                0.889,
                0.910,
                0.847,
                0.696,
                0.693,
            ],
            "ROC-AUC": [
                0.958,
                0.908,
                0.839,
                0.747,
                0.716,
            ],
        }
    )

    st.dataframe(
        benchmark_data,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# MODULE 4: RISK ASSESSMENT ENGINE
# =========================================================

elif page == "🩺 Risk Assessment Engine":

    st.title(
        "🩺 Patient Risk Assessment"
    )

    st.caption(
        "Input clinical biomarkers to estimate "
        "10-year Coronary Heart Disease probability."
    )

    if model is None:

        st.error(
            "Model artifacts are uninitialized. "
            "Check that your model file is saved correctly."
        )

        st.stop()

    # -----------------------------------------------------
    # PATIENT INPUT FORM
    # -----------------------------------------------------

    with st.form(
        "risk_assessment_form"
    ):

        st.subheader(
            "1. Demographics & Behavioral Parameters"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            age = st.number_input(
                "Age (Years)",
                min_value=18,
                max_value=100,
                value=52,
            )

        with c2:

            sex = st.selectbox(
                "Sex",
                ["Female", "Male"],
            )

        with c3:

            education = st.selectbox(
                "Education Level",
                [1, 2, 3, 4],
                help=(
                    "1: High School, "
                    "2: Some College, "
                    "3: Degree, "
                    "4: Advanced"
                ),
            )

        with c4:

            is_smoking = st.selectbox(
                "Current Smoker",
                ["No", "Yes"],
            )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            cigs_per_day = st.number_input(
                "Cigarettes / Day",
                min_value=0,
                max_value=100,
                value=(
                    0
                    if is_smoking == "No"
                    else 15
                ),
            )

        with c2:

            diabetes = st.selectbox(
                "Diabetic",
                ["No", "Yes"],
            )

        with c3:

            bp_meds = st.selectbox(
                "On Anti-Hypertensive Meds",
                ["No", "Yes"],
            )

        with c4:

            stroke = st.selectbox(
                "Prior Stroke History",
                ["No", "Yes"],
            )

        st.subheader(
            "2. Medical History & Clinical Biomarkers"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            hypertension = st.selectbox(
                "Prevalent Hypertension",
                ["No", "Yes"],
            )

        with c2:

            tot_chol = st.number_input(
                "Total Cholesterol (mg/dL)",
                min_value=100.0,
                max_value=600.0,
                value=220.0,
            )

        with c3:

            sys_bp = st.number_input(
                "Systolic BP (mmHg)",
                min_value=70.0,
                max_value=260.0,
                value=130.0,
            )

        with c4:

            dia_bp = st.number_input(
                "Diastolic BP (mmHg)",
                min_value=40.0,
                max_value=160.0,
                value=85.0,
            )

        c1, c2, c3 = st.columns(3)

        with c1:

            bmi = st.number_input(
                "Body Mass Index (BMI)",
                min_value=10.0,
                max_value=60.0,
                value=26.5,
            )

        with c2:

            heart_rate = st.number_input(
                "Heart Rate (bpm)",
                min_value=40.0,
                max_value=200.0,
                value=75.0,
            )

        with c3:

            glucose = st.number_input(
                "Glucose Level (mg/dL)",
                min_value=40.0,
                max_value=500.0,
                value=85.0,
            )

        st.divider()

        submit_btn = st.form_submit_button(
            "⚡ Evaluate Cardiovascular Risk",
            type="primary",
            use_container_width=True,
        )

    # -----------------------------------------------------
    # NEW PREDICTION
    # -----------------------------------------------------

    if submit_btn:

        with st.spinner(
            "Analyzing clinical biomarkers "
            "and extracting risk patterns..."
        ):

            time.sleep(1.5)

            input_payload = pd.DataFrame(
                [
                    {
                        "age": age,
                        "education": education,
                        "sex": (
                            1
                            if sex == "Male"
                            else 0
                        ),
                        "is_smoking": (
                            1
                            if is_smoking == "Yes"
                            else 0
                        ),
                        "cigsPerDay": cigs_per_day,
                        "BPMeds": (
                            1
                            if bp_meds == "Yes"
                            else 0
                        ),
                        "prevalentStroke": (
                            1
                            if stroke == "Yes"
                            else 0
                        ),
                        "prevalentHyp": (
                            1
                            if hypertension == "Yes"
                            else 0
                        ),
                        "diabetes": (
                            1
                            if diabetes == "Yes"
                            else 0
                        ),
                        "totChol": tot_chol,
                        "sysBP": sys_bp,
                        "diaBP": dia_bp,
                        "BMI": bmi,
                        "heartRate": heart_rate,
                        "glucose": glucose,
                    }
                ]
            )

            try:

                probability = float(
                    model.predict_proba(
                        input_payload
                    )[0][1]
                )

            except Exception as e:

                st.error(
                    "Model prediction failed."
                )

                st.caption(
                    f"Technical details: {str(e)}"
                )

                st.stop()

            is_high_risk = (
                probability >= threshold
            )

        # ---------------------------------------------
        # SAVE PREDICTION TO SESSION STATE
        # ---------------------------------------------

        st.session_state.last_probability = (
            probability
        )

        st.session_state.last_prediction = (
            int(is_high_risk)
        )

        st.session_state.last_is_high_risk = (
            is_high_risk
        )

        st.session_state.last_patient_data = (
            input_payload.copy()
        )

        # ---------------------------------------------
        # BUILD GEMINI CONTEXT
        # ---------------------------------------------

        st.session_state.patient_context = (
            build_patient_context(
                age=age,
                sex=sex,
                education=education,
                is_smoking=is_smoking,
                cigs_per_day=cigs_per_day,
                diabetes=diabetes,
                bp_meds=bp_meds,
                stroke=stroke,
                hypertension=hypertension,
                tot_chol=tot_chol,
                sys_bp=sys_bp,
                dia_bp=dia_bp,
                bmi=bmi,
                heart_rate=heart_rate,
                glucose=glucose,
                probability=probability,
                threshold_value=threshold,
            )
        )

        # ---------------------------------------------
        # RESET CHAT FOR NEW PREDICTION
        # ---------------------------------------------

        reset_cardio_chat()

    # -----------------------------------------------------
    # SHOW LAST PREDICTION
    # -----------------------------------------------------

    if st.session_state.last_probability is not None:

        render_risk_result(
            probability=(
                st.session_state.last_probability
            ),
            is_high_risk=(
                st.session_state.last_is_high_risk
            ),
            threshold_value=threshold,
            animation_warning=lottie_warning,
            animation_success=lottie_success,
        )

        st.divider()

        st.warning(
            """
            ⚠️ **Academic / Medical Disclaimer**

            This application is developed for academic,
            research, and educational demonstration purposes.

            The machine learning output is an estimated
            probability generated by the trained model.
            It is not a certified medical diagnosis and
            should not replace evaluation by a qualified
            healthcare professional.

            CardioAI provides general educational information
            and should not be used for diagnosis or treatment.
            """
        )

        render_cardio_ai()

    else:

        st.info(
            "Enter the patient's clinical information "
            "and click **Evaluate Cardiovascular Risk** "
            "to generate the ML prediction and activate "
            "the CardioAI assistant."
        )

        st.divider()

        st.subheader(
            "🤖 CardioAI"
        )

        st.caption(
            "CardioAI becomes fully contextual after "
            "generating a patient prediction."
        )

        if not os.getenv(
            GEMINI_API_KEY_NAME
        ):

            st.warning(
                "Gemini API key is not configured."
            )

        else:

            st.success(
                "Gemini connection is ready. "
                "Generate a prediction to provide "
                "patient context to CardioAI."
            )


# =========================================================
# MODULE 5: DOCUMENTATION
# =========================================================

elif page == "ℹ️ Documentation":

    st.title(
        "ℹ️ Technical Documentation"
    )

    st.markdown(
        """
        ### Project Architecture

        This platform combines a supervised machine
        learning model with a generative AI assistant
        for cardiovascular risk education.

        ### End-to-End Architecture

        **Patient Input**

        ↓

        **Preprocessing & Feature Encoding**

        ↓

        **Random Forest Classifier**

        ↓

        **10-Year CHD Risk Probability**

        ↓

        **CardioAI / Gemini**

        ↓

        **Natural Language Explanation & Health Education**

        ---

        ### Model Specifications

        * **Algorithm**:
          Random Forest Classifier

        * **Implementation**:
          `sklearn.ensemble.RandomForestClassifier`

        * **Hyperparameters**:
          `n_estimators=120`,
          `max_depth=12`,
          `random_state=42`

        * **Decision Calibration**:
          Custom probability threshold

        ---

        ### CardioAI Responsibilities

        CardioAI is used for:

        * Explaining cardiovascular diseases.
        * Explaining common cardiovascular risk factors.
        * Explaining the machine learning prediction.
        * Providing general cardiovascular health education.
        * Answering natural-language questions.

        CardioAI does **not**:

        * Recalculate the ML prediction.
        * Modify the ML prediction.
        * Diagnose the user.
        * Prescribe medication or dosage.
        * Replace a healthcare professional.

        ---

        ### Primary Biomarkers Used

        | Category | Features |
        | :--- | :--- |
        | **Demographics** | Age, Sex, Education Level |
        | **Behavioral** | Smoking Status, Cigarettes / Day |
        | **Medical History** | Prevalent Stroke, Prevalent Hypertension, Diabetes, Anti-Hypertensive Medications |
        | **Clinical Metrics** | Total Cholesterol, Systolic BP, Diastolic BP, BMI, Heart Rate, Glucose |

        ---

        ### Academic Disclaimer

        This system is developed for academic evaluation,
        machine learning demonstration, and educational
        purposes only.

        The prediction is an estimated model output and
        must not be interpreted as a medical diagnosis.
        """
    )
