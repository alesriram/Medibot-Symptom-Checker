"""
MediBot – Unified Predictor Service (Hybrid: SentenceTransformer + TF-IDF fallback)
"""

import logging
import re

log = logging.getLogger(__name__)

CONVERSATIONAL_TAGS = {
    "greeting", "goodbye", "work", "who", "Thanks",
    "joke", "name", "age", "gender", "not_understand", "center"
}

DISCLAIMER = (
    "This is an AI assistant, not a licensed doctor. "
    "Always consult a qualified healthcare professional for medical advice."
)

# Confidence thresholds — ST scores are generally higher than TF-IDF
# so we use separate thresholds per backend.
_THRESHOLDS = {
    "sentence_transformer": {"conv": 0.30, "medical": 0.35},
    "tfidf":                {"conv": 0.10, "medical": 0.15},
    "tfidf_fallback":       {"conv": 0.10, "medical": 0.15},
}


def predict(text: str) -> dict:
    text = re.sub(r"<[^>]+>", "", text).strip()
    if not text:
        return _not_understood("Please type something so I can help.")

    try:
        from app.services.embedder import EmbedderClassifier
        clf = EmbedderClassifier.get()

        # New signature: (tag, score, alternatives, method)
        tag, score, alternatives, method = clf.predict(text)

        if tag is not None and method != "none":
            thresholds = _THRESHOLDS.get(method, _THRESHOLDS["tfidf"])
            threshold = thresholds["conv"] if tag in CONVERSATIONAL_TAGS else thresholds["medical"]

            if score >= threshold:
                intent_data = clf.get_intent_data(tag)
                if intent_data:
                    return _build_response(tag, score, intent_data, alternatives, method=method)

            if alternatives:
                alt_str = ", ".join(f"{t} ({p}%)" for t, p in alternatives)
                return {
                    "tag": "not_understand",
                    "response": [
                        "not_understand",
                        f"Possible matches: {alt_str}. Can you describe your symptoms more specifically?"
                    ],
                    "confidence": score,
                    "method": method,
                    "alternatives": alternatives,
                }

    except Exception as exc:
        log.error("Predict failed: %s", exc, exc_info=True)

    return _not_understood("Service temporarily unavailable. Please try again.")


def _build_response(tag, score, intent_data, alternatives, method):
    import random
    if tag in CONVERSATIONAL_TAGS:
        resp = intent_data.get("responses", "")
        text = random.choice(resp) if isinstance(resp, list) else resp
        return {
            "tag": tag,
            "response": [tag, text],
            "confidence": score,
            "method": method,
            "alternatives": alternatives,
        }
    else:
        responses   = intent_data.get("responses", "No description available.")
        precaution  = intent_data.get("Precaution", "Consult a healthcare professional.")
        description = responses if isinstance(responses, str) else random.choice(responses)
        conf_str    = f"Confidence: {round(score * 100)}%"
        return {
            "tag": tag,
            "response": [tag, tag, description, precaution, conf_str, DISCLAIMER],
            "confidence": score,
            "method": method,
            "alternatives": alternatives,
        }


def _not_understood(msg):
    return {
        "tag": "not_understand",
        "response": ["not_understand", msg],
        "confidence": 0.0,
        "method": "none",
        "alternatives": [],
    }
