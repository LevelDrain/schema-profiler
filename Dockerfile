FROM python:3.11-slim

# GUI表示に必要なパッケージをWSL2向けに修正
RUN apt-get update && apt-get install -y \
    python3-tk \
    libglib2.0-0 \
    libgl1 \
    libgl1-mesa-dri \
    build-essential \
    && pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["/bin/bash"]

