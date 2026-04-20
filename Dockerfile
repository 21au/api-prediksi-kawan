FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN grep -v "pygrowup" requirements.txt > req_modern.txt && \
    pip install --no-cache-dir -r req_modern.txt

RUN pip install --no-cache-dir six && \
    pip install --no-cache-dir --no-build-isolation pygrowup==0.8.2

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]