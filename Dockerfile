# Docker image for the model serving API
FROM python:3.11-slim

WORKDIR /app

# Libraries are installed first, so this layer is cached between builds
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# API code and trained model
COPY api/ api/
COPY models/model.joblib models/model.joblib

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
