"""
MediBot – Unified Predictor Service (Hybrid: SentenceTransformer + TF-IDF fallback)

FIX: Added symptom keyword anchoring to prevent misclassification
     (e.g. "eye pain + redness" should NOT match Type 2 Diabetes just
      because both share "blurred vision").
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

# Raised medical threshold from 0.35 -> 0.42 to reduce false positives.
_THRESHOLDS = {
    "sentence_transformer": {"conv": 0.30, "medical": 0.42},
    "tfidf":                {"conv": 0.10, "medical": 0.18},
    "tfidf_fallback":       {"conv": 0.10, "medical": 0.18},
}

# ---------------------------------------------------------------------------
# Symptom-anchor rules: if the user message contains ANY of the listed
# keywords, BOOST preferred tags and PENALISE non-preferred tags.
# This stops "eye pain redness" matching Type 2 Diabetes because of
# the shared "blurred vision" signal.
# ---------------------------------------------------------------------------
_ANCHOR_RULES = [
    ({"eye", "eyes"},
        ["Conjunctivitis", "Glaucoma", "Uveitis", "Allergies", "Migraine",
         "Dizziness", "Stroke", "Multiple Sclerosis", "Hypertension", "Hypoglycemia"]),
    ({"blurr"},
        ["Conjunctivitis", "Glaucoma", "Uveitis", "Hypertension", "Migraine",
         "Type 2 Diabetes", "Hypoglycemia", "Stroke", "Multiple Sclerosis"]),
    ({"pink eye", "conjunctiv"},   ["Conjunctivitis"]),
    ({"glaucoma"},                 ["Glaucoma"]),
    ({"fever", "chills"},
        ["Fever", "Malaria", "Dengue", "Typhoid", "Meningitis",
         "Common Cold", "Tuberculosis", "Pneumonia", "Influenza"]),
    ({"chest pain", "chest tightness"},
        ["Heart Attack", "Asthma", "Bronchial Asthma",
         "GERD (Gastroesophageal Reflux Disease)", "Pulmonary Embolism",
         "Pneumothorax", "Anxiety", "Atrial Fibrillation"]),
    ({"rash", "skin rash"},
        ["Allergies", "Psoriasis", "Eczema", "Acne", "Impetigo",
         "Urticaria", "Drug Reaction", "Dengue", "Chicken pox", "Lupus",
         "Fungal Infection"]),
    ({"joint pain", "joint ache"},
        ["Rheumatoid Arthritis", "Arthritis", "Osteoarthritis", "Gout",
         "Dengue", "Lupus", "Fibromyalgia"]),
    ({"stomach pain", "abdominal pain", "belly pain"},
        ["Gastroenteritis", "Appendicitis", "Gallstones", "Pancreatitis",
         "IBS", "GERD (Gastroesophageal Reflux Disease)", "Food Poisoning",
         "Peptic ulcer disease", "Ovarian Cancer", "Ovarian Cyst", "Kidney Stones"]),
    ({"urination", "urinating", "urine"},
        ["Urinary Tract Infection (UTI)", "Type 2 Diabetes",
         "Chronic Kidney Disease", "Kidney Failure", "Kidney Stones"]),
]

_ANCHOR_BOOST   = 0.15
_ANCHOR_PENALTY = 0.20


def _apply_anchor_rules(text: str, intent_scores: dict) -> dict:
    """
    Apply keyword anchoring: boost preferred tags, penalise irrelevant ones.
    """
    text_lower = text.lower()
    adjusted = dict(intent_scores)

    for keywords, preferred_tags in _ANCHOR_RULES:
        if any(kw in text_lower for kw in keywords):
            preferred_set = set(preferred_tags)
            for tag in adjusted:
                if tag in CONVERSATIONAL_TAGS:
                    continue
                if tag in preferred_set:
                    adjusted[tag] = min(1.0, adjusted[tag] + _ANCHOR_BOOST)
                else:
                    adjusted[tag] = max(0.0, adjusted[tag] - _ANCHOR_PENALTY)

    return adjusted


_NAME_PATTERNS = re.compile(
    r"^\s*(?:my\s+name\s+is|i\s+am|i'm|call\s+me|this\s+is)\s+([a-zA-Z]{2,30})\s*[.!]?\s*$",
    re.IGNORECASE
)

def predict(text: str) -> dict:
    text = re.sub(r"<[^>]+>", "", text).strip()
    if not text:
        return _not_understood("Please type something so I can help.")

    # ── Name introduction detection ──
    name_match = _NAME_PATTERNS.match(text)
    if name_match:
        user_name = name_match.group(1).capitalize()
        return {
            "tag": "name",
            "response": ["name", f"Nice to meet you, {user_name}! 😊 How can I help you today? Please describe your symptoms."],
            "confidence": 1.0,
            "method": "rule",
            "alternatives": [],
        }

    try:
        from app.services.embedder import EmbedderClassifier
        clf = EmbedderClassifier.get()

        tag, score, alternatives, method = clf.predict(
            text, apply_anchor_fn=_apply_anchor_rules
        )

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
                        f"I couldn't pinpoint one condition. Possible matches: {alt_str}. "
                        "Could you describe your symptoms in more detail? For example, mention "
                        "where the pain is, how long it's been, and any other symptoms."
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
