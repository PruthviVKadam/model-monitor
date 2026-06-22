# Serve the Streamlit dashboard on Hugging Face Spaces (Docker SDK). Python 3.12 for broad wheels.
FROM python:3.12-slim

# libgomp1 = XGBoost's OpenMP runtime (booster.predict needs it; absent from python:slim).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableXsrfProtection=false"]
