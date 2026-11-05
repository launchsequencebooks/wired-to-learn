# Package the service as a container image:
#   docker build -t clothing-classifier .
#   docker run -p 8000:8000 clothing-classifier
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serve.py resnet20_int8.onnx ./
EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
