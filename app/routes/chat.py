"""MediBot – Chat, predict, triage, severity, RAG, hospital routes."""

import json
import math
import re
import datetime
import os

from flask import Blueprint, request, jsonify, session

from app.services.predictor import predict as service_predict
from app.services.severity  import get_severity, get_doctor, is_emergency
from database import save_chat

chat_bp = Blueprint("chat", __name__)

CONVERSATIONAL_TAGS = {
    "greeting", "goodbye", "work", "who", "Thanks",
    "joke", "name", "age", "gender", "not_understand", "center"
}

DISCLAIMER = (
    "This is an AI assistant, not a licensed doctor. "
    "Always consult a qualified healthcare professional for medical advice."
)

INTENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "intents.json")


@chat_bp.post("/predict")
def predict():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"answer": ["not_understand", "Invalid request format."]}), 400

        text = data.get("message", "").strip()
        text = re.sub(r"<[^>]+>", "", text)
        if not text:
            return jsonify({"answer": ["not_understand", "Please enter a message."]}), 400

        if is_emergency(text):
            return jsonify({"answer": ["emergency", "Emergency detected"], "emergency": True})

        result = service_predict(text)
        tag      = result["tag"]
        response = result["response"]
        method   = result["method"]

        severity = None
        doctor   = None
        if tag not in CONVERSATIONAL_TAGS and tag != "center":
            disease = response[1] if len(response) > 1 else tag
            severity = get_severity(disease)
            doctor   = get_doctor(disease)

        payload = {
            "answer": response,
            "confidence": round(result["confidence"] * 100),
            "method": method,
        }
        if severity: payload["severity"] = severity
        if doctor:   payload["doctor"]   = doctor
        if result["alternatives"]:
            payload["alternatives"] = result["alternatives"]

        return jsonify(payload)

    except Exception as exc:
        import traceback
        from flask import current_app
        current_app.logger.error("Predict error: %s\n%s", exc, traceback.format_exc())
        return jsonify({"answer": ["not_understand", "Server error. Please try again later."]}), 500


@chat_bp.post("/rag_context")
def rag_context():
    try:
        data        = request.get_json(silent=True) or {}
        disease     = data.get("disease", "")
        symptoms    = data.get("symptoms", "")
        description = data.get("description", "")
        precaution  = data.get("precaution", "")

        with open(INTENTS_PATH) as f:
            intents_data = json.load(f)

        rag_kb = ""
        for intent in intents_data["intents"]:
            if intent["tag"].lower() == disease.lower():
                rag_kb = (
                    f"Condition: {intent['tag']}\n"
                    f"Known Symptom Patterns: {', '.join(intent.get('patterns', []))}\n"
                    f"Medical Description: {intent.get('responses', '')}\n"
                    f"Recommended Precautions: {intent.get('Precaution', '')}"
                )
                break

        system_prompt = (
            "You are MediBot's advanced medical AI assistant. You provide compassionate, "
            "accurate, patient-friendly medical information. You never diagnose definitively. "
            "You always recommend consulting a healthcare professional. "
            "Respond ONLY with valid JSON, no markdown, no backticks."
        )

        user_prompt = f"""Based on this medical knowledge:
{rag_kb}

User reported symptoms: {symptoms}
Detected condition: {disease}
Base description: {description}
Base precautions: {precaution}

Provide a JSON response with:
{{
  "enriched_description": "2-3 sentence patient-friendly explanation",
  "why_these_symptoms": "Why reported symptoms match this condition (1 sentence)",
  "immediate_steps": ["3-4 immediate action steps"],
  "when_to_see_doctor": "Specific urgency guidance",
  "lifestyle_tips": ["3 lifestyle recommendations"],
  "doctor_specialization": "Best specialist type",
  "estimated_recovery": "Typical recovery if treated",
  "red_flags": ["2-3 warning signs that need emergency care"],
  "prevention": "Key prevention tip (1 sentence)"
}}"""

        return jsonify({
            "ok": True,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "disease": disease,
            "severity": get_severity(disease),
            "doctor": get_doctor(disease),
        })
    except Exception as exc:
        return jsonify({"ok": False, "msg": str(exc)}), 500


@chat_bp.post("/triage")
def triage():
    try:
        data         = request.get_json(silent=True) or {}
        symptoms_list = data.get("symptoms", [])
        if not symptoms_list:
            return jsonify({"ok": False, "msg": "Symptoms required"}), 400

        with open(INTENTS_PATH) as f:
            intents_data = json.load(f)

        relevant = []
        for intent in intents_data["intents"]:
            if intent["tag"] in CONVERSATIONAL_TAGS:
                continue
            patterns = [p.lower() for p in intent.get("patterns", [])]
            matches  = sum(1 for s in symptoms_list if any(s.lower() in p for p in patterns))
            if matches > 0:
                sev = get_severity(intent["tag"])
                relevant.append({
                    "tag":         intent["tag"],
                    "matches":     matches,
                    "description": intent.get("responses", ""),
                    "precaution":  intent.get("Precaution", ""),
                    "severity":    sev,
                    "doctor":      get_doctor(intent["tag"]),
                })

        relevant.sort(key=lambda x: (x["matches"], x["severity"]["urgency"]), reverse=True)
        symptoms_text = " ".join(symptoms_list).lower()

        return jsonify({
            "ok": True,
            "triage": {
                "symptoms":        symptoms_list,
                "top_conditions":  relevant[:5],
                "emergency":       is_emergency(symptoms_text),
                "overall_severity": relevant[0]["severity"] if relevant else get_severity(""),
            },
        })
    except Exception as exc:
        return jsonify({"ok": False, "msg": str(exc)}), 500


@chat_bp.post("/severity_predict")
def severity_predict():
    try:
        data     = request.get_json(silent=True) or {}
        symptoms = data.get("symptoms", "")
        vitals   = data.get("vitals", {})

        score, flags = 0, []

        temp = vitals.get("temp")
        hr   = vitals.get("hr")
        spo2 = vitals.get("spo2")

        if temp:
            temp = float(temp)
            if temp > 103:  score += 3; flags.append("Very high fever >103°F — seek care")
            elif temp > 100.4: score += 1; flags.append("Fever detected")
            elif temp < 96: score += 2; flags.append("Dangerously low temperature")

        if hr:
            hr = int(hr)
            if hr > 120:  score += 3; flags.append("Severe tachycardia (HR >120)")
            elif hr > 100: score += 1; flags.append("Elevated heart rate")
            elif hr < 50:  score += 2; flags.append("Bradycardia (HR <50)")

        if spo2:
            spo2 = int(spo2)
            if spo2 < 90:  score += 5; flags.append("Critical oxygen level <90% — EMERGENCY")
            elif spo2 < 95: score += 2; flags.append("Low oxygen saturation <95%")

        critical_symp = ["chest pain", "can't breathe", "difficulty breathing",
                         "unconscious", "seizure", "severe bleeding", "paralysis", "stroke"]
        moderate_symp = ["high fever", "vomiting", "severe headache", "confusion",
                         "weakness", "dizziness", "abdominal pain", "fainting"]

        for s in critical_symp:
            if s in symptoms.lower(): score += 3; flags.append(f"Critical symptom: {s}")
        for s in moderate_symp:
            if s in symptoms.lower(): score += 1

        if score >= 6:
            tier = {"level": "emergency", "label": "EMERGENCY", "color": "#7f1d1d",
                    "bg": "#fef2f2", "action": "Call 112 NOW or go to ER immediately", "score": score}
        elif score >= 4:
            tier = {"level": "urgent", "label": "URGENT", "color": "#dc2626",
                    "bg": "#fff5f5", "action": "See a doctor TODAY", "score": score}
        elif score >= 2:
            tier = {"level": "moderate", "label": "MODERATE", "color": "#d97706",
                    "bg": "#fffbeb", "action": "See a doctor within 24-48 hours", "score": score}
        else:
            tier = {"level": "mild", "label": "MILD", "color": "#16a34a",
                    "bg": "#f0fdf4", "action": "Rest, monitor symptoms, stay hydrated", "score": score}

        return jsonify({"ok": True, "severity": tier, "flags": flags, "score": score})
    except Exception as exc:
        return jsonify({"ok": False, "msg": str(exc)}), 500


@chat_bp.post("/log_symptom")
def log_symptom():
    data = request.get_json(silent=True) or {}
    if "history" not in session:
        session["history"] = []
    session["history"].append({
        "time":     datetime.datetime.now().isoformat(),
        "symptom":  data.get("symptom", "")[:500],
        "result":   data.get("result", "")[:1000],
        "severity": data.get("severity", ""),
    })
    session.modified = True
    username = session.get("username", "guest")
    save_chat(username, data.get("symptom", ""), data.get("result", ""), data.get("severity", ""))
    return jsonify({"status": "ok"})


@chat_bp.get("/history")
def get_history():
    return jsonify(session.get("history", []))


@chat_bp.get("/hospitals")
def hospitals():
    try:
        from chat import centres
        return jsonify({"answer": centres()})
    except Exception:
        return jsonify({"answer": ["not_understand", "Hospital lookup unavailable."]})


@chat_bp.post("/hospitals_nearby")
def hospitals_nearby():
    data = request.get_json(silent=True) or {}
    lat  = data.get("lat")
    lng  = data.get("lng")
    if not lat or not lng:
        return jsonify({"ok": False, "msg": "Location required"}), 400

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1; dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    centers_path = os.path.join(os.path.dirname(__file__), "..", "..", "medical_centers.json")
    with open(centers_path) as f:
        centers = json.load(f)["intents"]

    distances = []
    for c in centers:
        loc = c.get("location", [])
        if len(loc) >= 2:
            dist = haversine(lat, lng, loc[0], loc[1])
            distances.append({"name": c["tag"], "dist": round(dist, 2), "address": c.get("Address", "")})

    distances.sort(key=lambda x: x["dist"])
    return jsonify({"ok": True, "hospitals": distances[:5]})
