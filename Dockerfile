FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by lxml and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy production dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during image build
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Copy project files
COPY backend ./backend
COPY src ./src

# Create directories that will be mounted later
RUN mkdir -p /app/mlruns

# Python import path
ENV PYTHONPATH=/app/src:/app

# MLflow tracking database
ENV MLFLOW_TRACKING_URI=sqlite:///./mlflow.db

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]