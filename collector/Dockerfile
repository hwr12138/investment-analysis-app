# Built from the repository root: `gcloud builds submit --config collector/cloudbuild.yaml .`
# thetadata needs Python 3.12 or later; the Streamlit app does not import it.
FROM python:3.12-slim

WORKDIR /job
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY collector/requirements.txt collector/requirements.txt
RUN pip install --no-cache-dir -r collector/requirements.txt

# Only what the collector imports: the options arithmetic, the store layout,
# and the package markers above them. Nothing that imports Streamlit.
COPY app/__init__.py app/__init__.py
COPY app/options_layout.py app/options_layout.py
COPY app/logic/__init__.py app/logic/__init__.py
COPY app/logic/options app/logic/options
COPY collector collector

ENTRYPOINT ["python", "-m", "collector.main"]
CMD ["daily"]
