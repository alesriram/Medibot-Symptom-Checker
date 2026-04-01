"""
MediBot – Hybrid Intent Classifier
Primary  : Sentence Transformers (all-MiniLM-L6-v2) — semantic similarity
Fallback : TF-IDF cosine similarity — keyword-based, zero-memory-overhead

Strategy:
  1. Try SentenceTransformer → if import works + model loads, use it.
  2. If SentenceTransformer unavailable (OOM / missing package), fall back to TF-IDF.
  3. predictor.py gets a `method` field ("sentence_transformer" | "tfidf") so the
     frontend / logs know which path was taken.

Memory note:
  all-MiniLM-L6-v2 is ~90MB on disk, ~170MB RAM. Render free plan has 512MB,
  so it fits — but only just. If you see OOM, set env var
  MEDIBOT_FORCE_TFIDF=true to skip ST entirely.
"""

import json
import logging
import os
import numpy as np

log = logging.getLogger(__name__)

INTENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "intents.json")

CONVERSATIONAL_TAGS = {
    "greeting", "goodbye", "work", "who", "Thanks",
    "joke", "name", "age", "gender", "not_understand", "center"
}

# Set MEDIBOT_FORCE_TFIDF=true to disable Sentence Transformers
_FORCE_TFIDF = os.environ.get("MEDIBOT_FORCE_TFIDF", "false").lower() == "true"


# ─── TF-IDF backend ──────────────────────────────────────────────────────────

class _TFIDFBackend:
    """Lightweight scikit-learn TF-IDF cosine similarity classifier."""

    def __init__(self, intents):
        from sklearn.feature_extraction.text import TfidfVectorizer
        patterns, tags = [], []
        for intent in intents:
            for p in intent.get("patterns", []):
                patterns.append(p)
                tags.append(intent["tag"])

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        self._matrix = self._vectorizer.fit_transform(patterns)
        self._tags = tags
        log.info("TF-IDF backend ready — %d patterns indexed.", len(patterns))

    def similarity(self, text: str):
        from sklearn.metrics.pairwise import cosine_similarity
        q = self._vectorizer.transform([text])
        sims = cosine_similarity(q, self._matrix)[0]
        return sims, self._tags

    @property
    def method(self):
        return "tfidf"


# ─── Sentence Transformer backend ────────────────────────────────────────────

class _STBackend:
    """Sentence Transformers (all-MiniLM-L6-v2) semantic similarity classifier."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, intents):
        from sentence_transformers import SentenceTransformer
        import torch

        log.info("Loading SentenceTransformer model '%s' ...", self.MODEL_NAME)
        self._model = SentenceTransformer(self.MODEL_NAME)
        self._model.eval()

        patterns, tags = [], []
        for intent in intents:
            for p in intent.get("patterns", []):
                patterns.append(p)
                tags.append(intent["tag"])

        self._tags = tags

        with torch.no_grad():
            embs = self._model.encode(
                patterns, convert_to_numpy=True,
                show_progress_bar=False, batch_size=64
            )

        # L2-normalise so dot-product == cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        self._embs = embs / norms
        log.info("SentenceTransformer backend ready — %d embeddings cached.", len(patterns))

    def similarity(self, text: str):
        import torch
        with torch.no_grad():
            q = self._model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        q = q / (np.linalg.norm(q) + 1e-10)
        sims = (self._embs @ q.T).flatten()
        return sims, self._tags

    @property
    def method(self):
        return "sentence_transformer"


# ─── Hybrid Classifier ───────────────────────────────────────────────────────

class EmbedderClassifier:
    """
    Singleton hybrid classifier.

    predict() returns (tag, score, alternatives, method)
    get_intent_data(tag) returns the raw intent dict.
    """
    _instance = None

    def __init__(self):
        self._intents = []
        self._backend = None          # _STBackend or _TFIDFBackend (primary)
        self._tfidf_fallback = None   # always built as safety net
        self._ready = False
        self._load()

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        try:
            with open(INTENTS_PATH, "r") as f:
                data = json.load(f)
            self._intents = data["intents"]
        except Exception as exc:
            log.error("Could not load intents.json: %s", exc)
            return

        # Always build TF-IDF first — guaranteed fallback
        try:
            self._tfidf_fallback = _TFIDFBackend(self._intents)
        except Exception as exc:
            log.error("TF-IDF backend failed: %s", exc)

        # Try Sentence Transformers unless explicitly disabled
        if not _FORCE_TFIDF:
            try:
                self._backend = _STBackend(self._intents)
                log.info("Primary backend: Sentence Transformers")
            except ImportError:
                log.warning(
                    "sentence-transformers not installed — falling back to TF-IDF. "
                    "Install it with: pip install sentence-transformers"
                )
            except Exception as exc:
                log.warning("SentenceTransformer init failed (%s) — using TF-IDF fallback.", exc)

        if self._backend is None:
            self._backend = self._tfidf_fallback
            log.info("Primary backend: TF-IDF (sentence-transformers unavailable).")

        self._ready = self._backend is not None

    def predict(self, text: str):
        """
        Returns:
            (tag, score, alternatives, method)
            tag          : str   — predicted intent tag
            score        : float — similarity score [0, 1]
            alternatives : list of (tag, pct_int) tuples
            method       : str   — 'sentence_transformer' | 'tfidf' | 'tfidf_fallback' | 'none'
        """
        if not self._ready:
            return None, 0.0, [], "none"

        primary_method = self._backend.method

        try:
            tag, score, alternatives = self._run_backend(self._backend, text)
            return tag, score, alternatives, primary_method

        except Exception as exc:
            log.warning(
                "Primary backend (%s) predict failed: %s — trying TF-IDF fallback.",
                primary_method, exc
            )

        # Fallback to TF-IDF if primary (ST) crashed mid-request
        if self._tfidf_fallback and self._backend is not self._tfidf_fallback:
            try:
                tag, score, alternatives = self._run_backend(self._tfidf_fallback, text)
                return tag, score, alternatives, "tfidf_fallback"
            except Exception as exc2:
                log.error("TF-IDF fallback also failed: %s", exc2)

        return None, 0.0, [], "none"

    @staticmethod
    def _run_backend(backend, text: str):
        sims, tags = backend.similarity(text)

        intent_scores = {}
        for sim, tag in zip(sims, tags):
            s = float(sim)
            if tag not in intent_scores or s > intent_scores[tag]:
                intent_scores[tag] = s

        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        best_tag, best_score = sorted_intents[0]

        alternatives = [
            (tag, round(score * 100))
            for tag, score in sorted_intents
            if tag not in CONVERSATIONAL_TAGS and score > 0.10
        ][:3]

        return best_tag, best_score, alternatives

    def get_intent_data(self, tag: str):
        for intent in self._intents:
            if intent["tag"] == tag:
                return intent
        return None

    @property
    def available(self) -> bool:
        return self._ready

    @property
    def backend_name(self) -> str:
        return self._backend.method if self._backend else "none"
