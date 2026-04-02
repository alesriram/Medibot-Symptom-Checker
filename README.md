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
* **Machine Learning:** PyTorch (BiLSTM - optional)
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
│   │   ├── embedder.py      # Hybrid NLP (ST + TF-IDF fallback)
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

---

### 4. Set environment variables

```bash
# Required
export MEDIBOT_SECRET_KEY="your-secret-key"

# Optional (important for low memory environments)
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
Sentence Transformer (MiniLM)
   │
   ├── High confidence → return result
   │
   └── Failure / Low memory
          ▼
     TF-IDF fallback
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

For free-tier deployment (512MB RAM):

```bash
MEDIBOT_FORCE_TFIDF=true
```

* Prevents memory overflow
* Ensures fast startup
* Keeps app stable

---

## 🎯 Future Improvements

* Add frontend UI (React)
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
