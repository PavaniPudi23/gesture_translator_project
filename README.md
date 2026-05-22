# 🖐️ Personalized Gesture Language Translator

A real-time AI-powered gesture language translator built using **FastAPI, MediaPipe, Transformer Encoder, and Few-Shot Learning**.
This project detects hand gestures through a webcam, extracts landmarks using MediaPipe, generates gesture embeddings using a Transformer Encoder, and classifies personalized signs using a Few-Shot Prototypical Network.

---

# 🚀 Features

✅ Real-time hand gesture recognition
✅ MediaPipe hand landmark extraction
✅ Transformer-based gesture embedding encoder
✅ Few-shot learning for personalized sign enrollment
✅ User profile management
✅ Real-time WebSocket streaming
✅ Temporal smoothing for stable predictions
✅ Text-to-Speech output
✅ FastAPI backend with REST APIs
✅ Custom sign enrollment using only 3–5 examples
✅ Active learning through correction feedback

---

# 🧠 Tech Stack

## Backend

* Python
* FastAPI
* WebSockets
* PyTorch
* MediaPipe
* OpenCV

## Machine Learning

* Transformer Encoder
* Prototypical Networks
* Few-Shot Learning

## Frontend

* HTML
* CSS
* JavaScript
* Jinja2 Templates

---

# 📂 Project Structure

```plaintext
gesture-language-translator/
│
├── backend/
│   ├── main.py
│   ├── models/
│   │   ├── gesture_encoder.py
│   │   ├── few_shot_classifier.py
│   │   └── mediapipe_extractor.py
│   │
│   ├── utils/
│   │   ├── user_profile.py
│   │   └── tts_engine.py
│   │
│   └── weights/
│       └── encoder.pth
│
├── frontend/
│   ├── templates/
│   │   └── index.html
│   │
│   └── static/
│       ├── css/
│       ├── js/
│       └── assets/
│
├── user_data/
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/gesture-language-translator.git
cd gesture-language-translator
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

```txt
fastapi
uvicorn[standard]
pydantic
python-multipart
numpy
opencv-python
mediapipe
torch
torchvision
torchaudio
jinja2
aiofiles
scikit-learn
```

---

# ▶️ Running the Project

```bash
uvicorn main:app --reload
```

Open in browser:

```plaintext
http://localhost:8000
```

---

# 🖐️ How It Works

## Step 1 — Hand Detection

MediaPipe extracts 21 hand landmarks from webcam frames.

## Step 2 — Feature Encoding

The Transformer Encoder converts landmarks into compact gesture embeddings.

## Step 3 — Few-Shot Learning

The Prototypical Network compares embeddings with user-enrolled gesture prototypes.

## Step 4 — Prediction

The closest prototype label is returned as the predicted sign.

## Step 5 — Temporal Smoothing

Predictions are stabilized using a sliding window majority vote.

---

# 📡 API Endpoints

| Method | Endpoint                 | Description       |
| ------ | ------------------------ | ----------------- |
| GET    | `/api/health`            | Health check      |
| POST   | `/api/profile/create`    | Create new user   |
| GET    | `/api/profile/{user_id}` | Get profile       |
| POST   | `/api/predict`           | Predict gesture   |
| POST   | `/api/enroll`            | Enroll new sign   |
| POST   | `/api/correct`           | Submit correction |
| GET    | `/api/stats/{user_id}`   | User statistics   |
| GET    | `/api/tts/{text}`        | Text-to-speech    |
