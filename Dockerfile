FROM python:3.10-slim

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app .

EXPOSE 5000

CMD ["python", "main.py"]
