FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "\
import mediapipe as mp; \
p = mp.solutions.pose.Pose(model_complexity=0); \
p.close(); \
print('MediaPipe model downloaded successfully')"

COPY . .

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 10000

CMD streamlit run app.py --server.port=${PORT:-10000} --server.address=0.0.0.0
