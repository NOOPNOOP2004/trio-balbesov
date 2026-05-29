# Dockerfile
# Для разработки / тестирования на видеофайлах.
# Прямой доступ к USB-камере из Docker на Windows — отдельный шаг (см. README).

FROM python:3.12-slim

# Системные зависимости для OpenCV, MediaPipe и видео
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    v4l-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходники
COPY people_counter_v2.py .

# Папки для данных
RUN mkdir -p logs models screenshots videos

# По умолчанию — запуск с видеофайлом (для тестирования без камеры)
# Для камеры: docker run --device /dev/video0 ... (Linux/WSL2 only)
CMD ["python", "people_counter_v2.py", "--help"]
