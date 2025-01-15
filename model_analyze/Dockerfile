FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY templates ./templates
COPY final_baro_model.h5 ./final_baro_model.h5
COPY extract_model_analysis.py ./extract_model_analysis.py

EXPOSE 8000
CMD ["uvicorn", "extract_model_analysis:app", "--host", "0.0.0.0", "--port", "8000"]