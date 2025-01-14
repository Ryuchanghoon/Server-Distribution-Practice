FROM python:3.9-slim

WORKDIR /app

# OpenCV, mediapipe 의존성
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY templates ./templates
COPY calculate_coordinate.py ./calculate_coordinate.py

EXPOSE 8000

CMD ["uvicorn", "calculate_coordinate:app", "--host", "0.0.0.0", "--port", "8000"]