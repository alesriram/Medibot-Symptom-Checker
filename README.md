# 🏥 MediBot AI – Intelligent Medical Chatbot

An intelligent medical chatbot that uses a **hybrid NLP system** combining semantic embeddings (**Sentence Transformers**) and traditional **TF-IDF** to deliver fast, accurate, and scalable intent detection.

Designed with **real-world deployment constraints** in mind, the system dynamically switches to a lightweight model in low-memory environments (e.g., Render free tier), ensuring reliability and performance.

---

## 🚀 Features

* 🔹 Hybrid NLP classifier (Sentence Transformers + TF-IDF fallback)
* 🔹 Automatic fallback for low-memory environments (512MB RAM safe)
* 🔹 Multi-endpoint REST API (chat, triage, severity prediction)
* 🔹 Location-based hospital search
* 🔹 Session history tracking
* 🔹 Modular Flask architecture (production-ready)
* 🔹 Optimized for cloud deployment (Render)

---

## 💡 Key Highlight

Implemented a **dynamic hybrid NLP pipeline**:

* Uses **Sentence Transformers (MiniLM)** for high semantic accuracy
* Automatically switches to **TF-IDF** if memory is limited or model fails
* Prevents crashes and ensures **stable performance on low-resource systems**
* Demonstrates real-world production engineering practices

---

## 🛠 Tech Stack

* **Backend:** Flask, Gunicorn
* **NLP:** Sentence Transformers (`all-MiniLM-L6-v2`), TF-IDF (scikit-learn)
* **Database:** SQLite / MySQL
* **Deployment:** Render

---

## 📂 Project Structure

```
medibot/
├── app/
│   ├── routes/
│   │   ├── auth.py
│   │   ├── admin.py
│   │   └── chat.py
│   ├── services/
│   │   ├── embedder.py      # Hybrid NLP (Sentence Transformers + TF-IDF fallback)
│   │   ├── predictor.py
│   │   └── severity.py
│   └── utils/
│       └── health.py
├── intents.json
├── main.py
├── database.py
├── requirements.txt
├── render.yaml
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/alesriram/Medibot-Symptom-Checker
cd medibot
```

---

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` does **not** include `sentence-transformers` by default because the Render free tier only provides **512MB RAM**, and the Sentence Transformer model (`all-MiniLM-L6-v2`) requires approximately **170MB RAM** on top of Flask, NLTK, and scikit-learn — which causes memory overflow on Render's free plan.
>
> On Render, the app automatically uses **TF-IDF mode** (set via `MEDIBOT_FORCE_TFIDF=true` in `render.yaml`), which is lightweight and stable.
>
> **To run locally with the full Sentence Transformer model**, install it separately:
>
> ```bash
> pip install sentence-transformers>=2.6.0
> ```
>
> Then make sure `MEDIBOT_FORCE_TFIDF` is set to `false` (or not set at all) in your local environment.

---

### 4. Set environment variables

```bash
# Required
export MEDIBOT_SECRET_KEY="your-secret-key"

# Set to false locally if you have installed sentence-transformers
# Set to true on Render free tier to avoid memory overflow
export MEDIBOT_FORCE_TFIDF=false

# Database (optional)
export MYSQL_HOST=localhost
export MYSQL_USER=root
export MYSQL_PASSWORD=password
export MYSQL_DB=medibot
```

---

### 5. Run the application

#### Development

```bash
python main.py
```

#### Production

```bash
gunicorn main:app --workers 1 --bind 0.0.0.0:5000
```

---

## 🧠 How the Hybrid Model Works

```
User Input
   │
   ▼
Sentence Transformer (MiniLM)        ← used locally (high accuracy)
   │
   ├── High confidence → return result
   │
   └── Failure / Low memory / FORCE_TFIDF=true
          ▼
     TF-IDF fallback                 ← used on Render free tier
          ▼
     Return best match
```

---

## 📡 API Endpoints

| Method | Endpoint            | Description            |
| ------ | ------------------- | ---------------------- |
| POST   | `/predict`          | Chat prediction        |
| POST   | `/triage`           | Multi-symptom triage   |
| POST   | `/rag_context`      | Build AI prompt        |
| POST   | `/severity_predict` | Severity scoring       |
| POST   | `/log_symptom`      | Store symptom data     |
| GET    | `/history`          | Retrieve chat history  |
| POST   | `/hospitals_nearby` | Nearby hospital search |
| GET    | `/health`           | Health check           |

---

## 📸 Sample Response

```json
{
  "intent": "fever",
  "confidence": 0.82,
  "method": "tfidf_fallback",
  "response": "You may have a fever. Stay hydrated and consult a doctor if symptoms persist."
}
```

---

## ☁️ Deployment (Render)

The app is configured for Render's free plan via `render.yaml`.

Key setting for free tier stability:

```bash
MEDIBOT_FORCE_TFIDF=true
```

* Skips Sentence Transformer model loading entirely
* Prevents memory overflow on 512MB RAM
* Ensures fast startup and stable performance
* TF-IDF alone provides reliable intent matching for most queries

---

## 🎯 Future Improvements

* Integrate real-time doctor APIs
* Improve medical dataset coverage
* Add multilingual support

---

## 👨‍💻 Ale Sriram

* GitHub: https://github.com/alesriram
* LinkedIn: https://www.linkedin.com/in/ale-sai-sriram-kumar-6035b0272/

---

## ⭐ Acknowledgment

This project demonstrates **practical NLP engineering**, focusing on balancing **accuracy vs performance** in real-world deployment scenarios.