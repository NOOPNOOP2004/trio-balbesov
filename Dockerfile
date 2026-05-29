FROM python:3.12-slim

# Установка системных зависимостей для OpenCV и MediaPipe
# libgl1-mesa-glx  — нужен OpenCV для декодирования видео
# libglib2.0-0     — MediaPipe / glib
# libsm6, libxext6, libxrender-dev — OpenCV runtime (headless тоже требует)
# libgomp1         — OpenMP для YOLO (параллельные вычисления)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости отдельным слоем — пересборка только при изменении requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Исходный код и модели
COPY people_counter_robust.py .
COPY yolo11n.pt .
COPY yolov8n.pt .

# Папки для логов, скриншотов и tflite-моделей (будут смонтированы через volumes)
RUN mkdir -p logs models screenshots

# ИСПРАВЛЕНО: headless-режим через переменную окружения.
# Скрипт читает HEADLESS=1 и пропускает cv2.imshow / cv2.waitKey.
# Для работы с видеофайлом — передай --source через docker-compose или CLI.
ENV HEADLESS=1

# ИСПРАВЛЕНО: запуск с видеофайлом по умолчанию, не с камерой.
# Камера (/dev/video0) в контейнере без явного проброса недоступна.
CMD ["python", "people_counter_robust.py", "--source", "video.mp4", "--no-multiscale"]
