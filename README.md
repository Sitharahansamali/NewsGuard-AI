# Fake News Detection AI

AI-powered fake news detection web application using Machine Learning, FastAPI, Streamlit, and MongoDB.

---

# Features

- Detect fake and real news
- Machine Learning based prediction
- FastAPI backend
- Streamlit frontend
- MongoDB database integration
- Model training using Jupyter Notebook
- Deployable on Render

---

# Technologies Used

- Python
- Anaconda
- Jupyter Notebook
- Scikit-learn
- FastAPI
- Streamlit
- MongoDB Atlas
- Render

---

# Project Structure

```text
NewsGuard_AI/
│
├── data/
├── notebooks/
├── backend/
├── frontend/
├── saved_models/
├── requirements.txt
└── README.md
```

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone <your-github-repository-link>
```

```bash
cd NewsGuard_AI
```

---

## 2. Create Conda Environment

```bash
conda create -n fake_news_ai python=3.14
```

Activate environment:

```bash
conda activate fake_news_ai
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/model_training.ipynb
```

---

# Run FastAPI Backend

Go to backend folder:

```bash
cd backend
```

Run server:

```bash
uvicorn main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

---

# Run Streamlit Frontend

Open new terminal.

Go to frontend folder:

```bash
cd frontend
```

Run Streamlit app:

```bash
streamlit run app.py
```

Frontend URL:

```text
http://localhost:8501
```

---

# MongoDB Setup

1. Create MongoDB Atlas account
2. Create cluster
3. Get connection string
4. Add MongoDB URI inside backend

MongoDB Website:

[MongoDB Atlas](https://www.mongodb.com/atlas/database?utm_source=chatgpt.com)

---

# Dataset

Use Kaggle Fake and Real News Dataset.

Download:

[Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?utm_source=chatgpt.com)

Place dataset files inside:

```text
data/
```

---

# Future Improvements

- BERT model integration
- Confidence score prediction
- Explainable AI
- URL-based fake news detection
- Multilingual support

---

# Deployment

Deploy project using:

- Render
- Streamlit Cloud

Render:

[Render](https://render.com?utm_source=chatgpt.com)

---

# Author

Sithara Hansamali  
Computer Science Undergraduate