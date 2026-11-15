# The API service on Cloud Run. Built from the repository root:
#   gcloud builds submit --config deploy/api/cloudbuild.yaml .
FROM python:3.13-slim

WORKDIR /srv
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# The package only: no streamlit_app.py, no tests, and .gcloudignore keeps
# .streamlit/ (the local secrets) out of the build context entirely.
COPY app app

# One worker: the data layer's caches are in-process, so a second worker would
# be a second, cold cache. Cloud Run's concurrency serves parallel requests
# from FastAPI's thread pool instead.
CMD exec uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
