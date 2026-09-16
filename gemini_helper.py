import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# =========================================================
# GEMINI MODEL
# =========================================================

MODEL_NAME = "gemini-3.6-flash"


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are CardioAI, a cardiovascular health education assistant
inside an academic machine learning application.

Your responsibilities:
- Explain cardiovascular diseases in simple language.
- Explain common cardiovascular risk factors.
- Provide general cardiovascular health education.
- Explain the machine learning prediction when prediction
  information is provided.
- Explain patient input values when relevant.
- Answer general cardiovascular health questions.

IMPORTANT SAFETY RULES:
- Never calculate, recalculate, or modify the machine
  learning risk score.
- The machine learning model is responsible for the
  numerical prediction.
- Treat the prediction as an estimate, not a medical diagnosis.
- Never diagnose a patient.
- Never prescribe medications, treatments, or dosages.
- Do not invent patient information.
- Do not assume missing patient information.
- Do not claim certainty about a medical condition.
- Encourage consultation with a qualified healthcare
  professional for personal medical decisions.
- If the user describes severe or emergency symptoms,
  advise seeking urgent medical care.

LANGUAGE RULES:
- Always respond in the same language used by the user.
- If the user asks in Arabic, respond in Arabic.
- If the user asks in English, respond in English.
- If the user mixes Arabic and English, respond naturally
  using the same style when appropriate.
- Do not translate the user's question unless explicitly requested.
- When responding in Arabic, use clear, simple, professional
  Arabic that is easy for a university student to understand.
- Keep medical terminology accurate even when simplifying
  the explanation.

STYLE RULES:
- Keep responses clear, practical, and reasonably concise.
- Explain technical or medical concepts in a simple way.
- Avoid unnecessary complexity.
- Do not use overly formal or difficult language.
"""


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_gemini_client():
    """Create the Gemini client using the environment API key."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. "
            "Please configure your Gemini API key."
        )

    return genai.Client(api_key=api_key)


# =========================================================
# CONVERSATION BUILDER
# =========================================================

def build_conversation(history):
    """
    Convert the Streamlit chat history into readable
    conversation text for Gemini.
    """

    if not history:
        return "No previous conversation."

    conversation = []

    for message in history:

        role = message.get(
            "role",
            "user",
        )

        content = message.get(
            "content",
            "",
        )

        if role == "user":

            conversation.append(
                f"User: {content}"
            )

        elif role == "assistant":

            conversation.append(
                f"CardioAI: {content}"
            )

    return "\n".join(conversation)


# =========================================================
# GENERATE CARDIO AI RESPONSE
# =========================================================

def generate_cardio_response(
    user_message,
    chat_history=None,
    patient_context=None,
):
    """
    Generate a CardioAI response using Gemini.

    The Gemini model explains the ML prediction and
    provides educational information. It does not
    calculate or modify the ML risk score.
    """

    # -----------------------------------------------------
    # Create Gemini client
    # -----------------------------------------------------

    client = get_gemini_client()

    # -----------------------------------------------------
    # Safe defaults
    # -----------------------------------------------------

    if chat_history is None:
        chat_history = []

    if patient_context is None:
        patient_context = (
            "No current patient prediction "
            "is available."
        )

    # -----------------------------------------------------
    # Build previous conversation
    # -----------------------------------------------------

    conversation = build_conversation(
        chat_history
    )

    # -----------------------------------------------------
    # Build user prompt
    # -----------------------------------------------------

    prompt = f"""
CURRENT PATIENT / MODEL CONTEXT
--------------------------------

{patient_context}


PREVIOUS CONVERSATION
--------------------------------

{conversation}


CURRENT USER QUESTION
--------------------------------

{user_message}


INSTRUCTIONS
--------------------------------

1. Answer the user's current question directly.

2. Respond in the same language as the user's current
   question.

3. If the user asks in Arabic, answer in Arabic.
   Use clear and simple professional Arabic.

4. If the user asks in English, answer in English.

5. If the user asks about the machine learning prediction,
   explain only the prediction that has already been
   calculated by the ML model.

6. Do NOT calculate a new risk score.

7. Do NOT change, override, or reinterpret the numerical
   prediction produced by the ML model.

8. Do NOT invent missing patient information.

9. Provide general educational information rather than
   a diagnosis or personalized medical treatment.

10. If the user asks about emergency symptoms, recommend
    seeking urgent medical care.

11. Keep the answer practical and reasonably concise.
"""

    # -----------------------------------------------------
    # Generate Gemini response
    # -----------------------------------------------------

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        ),
    )

    # -----------------------------------------------------
    # Handle empty response
    # -----------------------------------------------------

    if not response.text:

        return (
            "I could not generate a response right now. "
            "Please try again."
        )

    # -----------------------------------------------------
    # Return response
    # -----------------------------------------------------

    return response.text
