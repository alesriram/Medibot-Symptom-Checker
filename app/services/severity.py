import json
import logging
import os

log = logging.getLogger(__name__)

INTENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "intents.json")

# ── Severity Mapping ─────────────────────────────────────
SEVERITY_MAP = {
    "cancer": 5, "hiv": 5, "aids": 5, "hiv/aids": 5,
    "heart attack": 5, "stroke": 5, "sepsis": 5,
    "paralysis (brain hemorrhage)": 5, "paralysis": 5,
    "hemorrhage": 5, "tuberculosis": 5, "pneumothorax": 5,
    "meningitis": 5, "ovarian cancer": 5,

    "pneumonia": 5, "dengue": 4, "malaria": 4, "typhoid": 4,
    "hepatitis a": 4, "hepatitis b": 4, "hepatitis c": 4,
    "hepatitis d": 4, "hepatitis e": 4, "alcoholic hepatitis": 4,
    "chronic kidney disease": 4, "lupus": 4, "multiple sclerosis": 4,

    "asthma": 3, "bronchial asthma": 3, "arthritis": 3,
    "rheumatoid arthritis": 3, "osteoarthritis": 3,
    "diabetes": 3, "type 2 diabetes": 3, "hypoglycemia": 3,
    "migraine": 3, "hypertension": 3, "copd": 3,
    "hypothyroidism": 3, "hyperthyroidism": 3,
    "atrial fibrillation": 3, "appendicitis": 3,
    "pancreatitis": 3, "gallstones": 3,
    "peptic ulcer disease": 3, "celiac disease": 3,
    "chronic cholestasis": 3, "chronic fatigue syndrome": 3,
    "fibromyalgia": 3, "ibs": 3,
    "kidney stones": 3, "urinary tract infection (uti)": 3,
    "varicose veins": 3, "anemia": 3, "depression": 3,
    "anxiety": 3, "gerd (gastroesophageal reflux disease)": 3,

    "common cold": 1, "fever": 2, "acne": 1,
    "allergies": 1, "allergy": 1, "eczema": 2,
    "psoriasis": 2, "drug reaction": 2, "impetigo": 2,
    "chicken pox": 2, "fungal infection": 2,
    "gastroenteritis": 2, "dizziness": 2, "vertigo": 2,
    "sinusitis": 2, "gout": 2,
    "dimorphic hemmorhoids(piles)": 2, "jaundice": 2,
    "ovarian cyst": 2,
}

URGENCY_LABELS = {
    5: {"level": "critical", "label": "Critical", "color": "#ef4444",
        "action": "Seek emergency care immediately", "urgency": 5},
    4: {"level": "high", "label": "High", "color": "#f97316",
        "action": "See a doctor today", "urgency": 4},
    3: {"level": "moderate", "label": "Moderate", "color": "#f59e0b",
        "action": "Consult a doctor within 24–48 hours", "urgency": 3},
    2: {"level": "mild", "label": "Mild", "color": "#84cc16",
        "action": "Monitor symptoms, rest and hydrate", "urgency": 2},
    1: {"level": "minimal", "label": "Minimal", "color": "#22c55e",
        "action": "Self-care at home; see doctor if persists", "urgency": 1},
}

EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "unconscious", "stroke", "heart attack", "severe bleeding",
    "not breathing", "seizure", "paralysis",
]

DOCTOR_MAP = {
    "heart": "Cardiologist", "cardiac": "Cardiologist",
    "atrial": "Cardiologist", "heart attack": "Cardiologist",
    "lung": "Pulmonologist", "asthma": "Pulmonologist",
    "copd": "Pulmonologist", "pneumonia": "Pulmonologist",
    "bronchial": "Pulmonologist", "pneumothorax": "Pulmonologist",
    "diabetes": "Endocrinologist", "thyroid": "Endocrinologist",
    "hypoglycemia": "Endocrinologist",
    "skin": "Dermatologist", "acne": "Dermatologist",
    "psoriasis": "Dermatologist", "eczema": "Dermatologist",
    "impetigo": "Dermatologist", "fungal": "Dermatologist",
    "arthritis": "Rheumatologist", "fibromyalgia": "Rheumatologist",
    "lupus": "Rheumatologist",
    "joint": "Orthopedist", "bone": "Orthopedist",
    "osteoarthritis": "Orthopedist",
    "depression": "Psychiatrist", "anxiety": "Psychiatrist",
    "multiple sclerosis": "Neurologist",
    "gastro": "Gastroenterologist", "stomach": "Gastroenterologist",
    "ulcer": "Gastroenterologist", "ibs": "Gastroenterologist",
    "gerd": "Gastroenterologist", "celiac": "Gastroenterologist",
    "pancreatitis": "Gastroenterologist", "gallstones": "Gastroenterologist",
    "liver": "Hepatologist", "hepatitis": "Hepatologist",
    "kidney": "Nephrologist", "urinary": "Urologist",
    "uti": "Urologist",
    "cancer": "Oncologist",
    "migraine": "Neurologist",
    "meningitis": "Neurologist",
    "hiv": "Infectious Disease Specialist",
    "malaria": "Infectious Disease Specialist",
    "typhoid": "Infectious Disease Specialist",
    "dengue": "Infectious Disease Specialist",
    "anemia": "Hematologist",
    "ovarian": "Gynecologist",
    "sinusitis": "ENT Specialist",
    "vertigo": "ENT Specialist",
}

# ── CLASS ───────────────────────────────────────────────
class SeverityClassifier:

    _instance = None

    def __init__(self):
        self._clf = None
        self._vectorizer = None
        self._ready = False
        self._intents = None  # lazy load

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_intents(self):
        try:
            with open(INTENTS_PATH) as f:
                return json.load(f).get("intents", [])
        except Exception as e:
            log.warning("Failed to load intents.json: %s", e)
            return []

    def _train(self):
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.feature_extraction.text import TfidfVectorizer

            X_texts, y_labels = [], []

            for intent in self._intents:
                tag = intent["tag"].lower()
                urgency = SEVERITY_MAP.get(tag, 1)

                for pattern in intent.get("patterns", []):
                    X_texts.append(pattern)
                    y_labels.append(urgency)

            if not X_texts:
                return

            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
            X = self._vectorizer.fit_transform(X_texts)

            self._clf = LogisticRegression(max_iter=500)
            self._clf.fit(X, y_labels)

            self._ready = True
            log.info("Severity model trained (%d samples)", len(X_texts))

        except Exception as e:
            log.warning("ML disabled, fallback active: %s", e)

    def predict_severity(self, disease_name: str) -> dict:
        text = disease_name.lower()

        # Lazy init (IMPORTANT)
        if self._intents is None:
            self._intents = self._load_intents()
            self._train()

        # Direct mapping
        if text in SEVERITY_MAP:
            return URGENCY_LABELS[SEVERITY_MAP[text]]

        # ML prediction
        if self._ready:
            try:
                X = self._vectorizer.transform([disease_name])
                pred = int(self._clf.predict(X)[0])
                return URGENCY_LABELS.get(pred, URGENCY_LABELS[1])
            except Exception as e:
                log.warning("ML prediction failed: %s", e)

        # Fallback
        return _keyword_severity(text)

    @property
    def available(self):
        return self._ready


# ── FALLBACK ────────────────────────────────────────────
def _keyword_severity(text: str) -> dict:
    HIGH = ["cancer", "hiv", "heart attack", "stroke", "sepsis"]
    MED = ["asthma", "diabetes", "migraine", "hypertension"]

    if any(k in text for k in HIGH):
        return URGENCY_LABELS[5]
    if any(k in text for k in MED):
        return URGENCY_LABELS[3]

    return URGENCY_LABELS[1]


# ── HELPERS ─────────────────────────────────────────────
def get_severity(disease_name: str) -> dict:
    return SeverityClassifier.get().predict_severity(disease_name)


def get_doctor(disease_name: str) -> str:
    dl = disease_name.lower()
    for key, spec in DOCTOR_MAP.items():
        if key in dl:
            return spec
    return "General Physician"


def is_emergency(text: str) -> bool:
    return any(kw in text.lower() for kw in EMERGENCY_KEYWORDS)