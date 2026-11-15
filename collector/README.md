# Options collector — deploy runbook

The 🎯 Options page reads a private GCS bucket; this job fills it. It runs on
**Cloud Run Jobs**, triggered nightly by **Cloud Scheduler**. ThetaData's free
tier is too slow to call from the page: it allows one request at a time, at
about 1–2.5 s of server time per trading day.

Everything below is run once, from the repository root, by someone signed in to
`gcloud`. Replace `PROJECT`, `BUCKET` and `REGION`. `us-central1` is a fine
default.

## 0. Tools

```bash
brew install --cask google-cloud-sdk
gcloud auth login
gcloud config set project PROJECT
```

## 1. APIs, bucket, image repository

```bash
gcloud services enable run.googleapis.com cloudscheduler.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
gcloud storage buckets create gs://BUCKET --location=REGION --uniform-bucket-level-access --public-access-prevention
gcloud artifacts repositories create options --repository-format=docker --location=REGION
```

## 2. The collector's service account

It can write the bucket and read its one secret. The API reads the bucket as its own account,
which `web/README.md` sets up (Deploy, step 3).

```bash
gcloud iam service-accounts create options-collector
gcloud storage buckets add-iam-policy-binding gs://BUCKET --member=serviceAccount:options-collector@PROJECT.iam.gserviceaccount.com --role=roles/storage.objectAdmin
```

## 3. The ThetaData key, in Secret Manager

```bash
printf '%s' 'YOUR_THETADATA_KEY' | gcloud secrets create thetadata-api-key --data-file=-
gcloud secrets add-iam-policy-binding thetadata-api-key --member=serviceAccount:options-collector@PROJECT.iam.gserviceaccount.com --role=roles/secretmanager.secretAccessor
```

## 4. Build and deploy the job

The build runs as its own service account, which can only push images and write
logs:

```bash
gcloud iam service-accounts create options-builder
gcloud projects add-iam-policy-binding PROJECT --member=serviceAccount:options-builder@PROJECT.iam.gserviceaccount.com --role=roles/artifactregistry.writer
gcloud projects add-iam-policy-binding PROJECT --member=serviceAccount:options-builder@PROJECT.iam.gserviceaccount.com --role=roles/logging.logWriter
gcloud storage buckets add-iam-policy-binding gs://PROJECT_cloudbuild --member=serviceAccount:options-builder@PROJECT.iam.gserviceaccount.com --role=roles/storage.objectViewer
```

The last binding lets the build read the source that `gcloud builds submit` uploads to
`PROJECT_cloudbuild`; without it the build fails before its first step. The bucket exists once
any build has been submitted. The API service (`deploy/api/`) builds the same way.

`.gcloudignore` keeps `.streamlit/` (your secrets) out of the upload.

```bash
gcloud builds submit --config collector/cloudbuild.yaml --substitutions=_REGION=REGION \
  --service-account=projects/PROJECT/serviceAccounts/options-builder@PROJECT.iam.gserviceaccount.com .
gcloud run jobs deploy options-collector \
  --image=REGION-docker.pkg.dev/PROJECT/options/collector:latest --region=REGION \
  --service-account=options-collector@PROJECT.iam.gserviceaccount.com \
  --set-secrets=THETADATA_API_KEY=thetadata-api-key:latest \
  --set-env-vars=OPTIONS_STORE_URI=gs://BUCKET \
  --task-timeout=24h --max-retries=0 --memory=2Gi --cpu=1
```

After that, a merge to `main` that changes the collector rebuilds its image and points the job
at it (`deploy/README.md`). To deploy by hand, rerun both commands.

## 5. Backfill the last year, once

This takes about 12–18 hours, because the free tier allows one request at a time.
Measured locally, each name-month took about 45 s of ThetaData time and 2 s of
computation. The job checkpoints each finished (name, month) in the manifest, so
if it stops for any reason, running the same command again picks up where it
left off. To see the page sooner, backfill a few names first with
`--args=backfill,--start,2025-10-24,--symbols,GLD,SPY,QQQ`; a later full run
skips what is already done.

```bash
gcloud run jobs execute options-collector --region=REGION --args=backfill,--start,2025-10-24
```

Then run one daily pass, which writes the latest full chains:

```bash
gcloud run jobs execute options-collector --region=REGION --args=daily --wait
```

## 6. The nightly schedule

ThetaData releases each session after its 17:15 ET report, and the free tier
serves it from the next day. The job catches up on every session it has not yet
stored, so the 07:00 trigger does nothing if the 01:30 run succeeded.

```bash
gcloud iam service-accounts create options-scheduler
gcloud run jobs add-iam-policy-binding options-collector --region=REGION --member=serviceAccount:options-scheduler@PROJECT.iam.gserviceaccount.com --role=roles/run.invoker
for spec in "options-nightly|30 1 * * 2-6" "options-retry|0 7 * * 2-6"; do
  name=${spec%%|*}; cron=${spec#*|}
  gcloud scheduler jobs create http "$name" --location=REGION --schedule="$cron" --time-zone=America/New_York \
    --uri="https://run.googleapis.com/v2/projects/PROJECT/locations/REGION/jobs/options-collector:run" --http-method=POST \
    --oauth-service-account-email=options-scheduler@PROJECT.iam.gserviceaccount.com
done
```

## 7. Point the API at the bucket

The API service is deployed with `OPTIONS_STORE_URI=gs://BUCKET` and reads it as its own service
account, so it needs no key file; `web/README.md` (Deploy) has both steps. Run locally, the API
reads `OPTIONS_STORE_URI` from the environment, or `OPTIONS_BUCKET` plus a `[gcp_service_account]`
table from `.streamlit/secrets.toml`, and otherwise your own application-default credentials.

## Rebuilding the screens

Each daily run also solves the night's chain into the tables the web app's Discover and Screen
tabs read (`latest/discover.parquet`, `latest/spreads.parquet`, `latest/expirations.parquet`), and
records the close they were solved from as `screens.session` in the manifest. The API refuses
tables from an older close than the manifest's, so a store a collector without this step last
wrote shows an error rather than yesterday's screens. To build them from what is stored, with no
ThetaData request (about 15 s), run **Execute with overrides** with the argument `screens`:

```bash
gcloud run jobs execute options-collector --region=REGION --args=screens --wait
```

The image must be rebuilt first (§4) for the job to know the mode.

## When a metric's definition changes

Every raw chain is kept in the bucket, so the history can be rebuilt without a
ThetaData request. Run **Execute with overrides** with the single argument
`recompute`, or `--args=recompute` from the command line. It takes about 0.1 s
per name-session, so roughly 40 minutes for a year of 90 names. It rebuilds the screens
afterwards, since the Discover table ranks against the history it restated.

## Running it locally

The job runs locally against a directory instead of a bucket. You need Python
3.12 or later, and the package's requirements from `collector/requirements.txt`.
Leave out `--symbols` to cover the whole universe.

```bash
export THETADATA_API_KEY=…
python -m collector.main backfill --start 2025-10-24 --symbols GLD,SPY --store file:///tmp/options-store
python -m collector.main daily --symbols GLD,SPY --store file:///tmp/options-store
OPTIONS_STORE_URI=file:///tmp/options-store EQUITI_AUTH=off python3 -m uvicorn app.api.main:app --port 8600
```

## Cost

| Item | Monthly cost |
|---|---|
| Cloud Run job (5–10 min a night) | within the free tier |
| Cloud Scheduler (2 of the 3 free jobs) | within the free tier |
| Secret Manager (one secret) | within the free tier |
| Cloud Storage (~2–3 GB a year of Parquet) | about $0.05 |
