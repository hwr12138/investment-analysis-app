# Equiti web (proof of concept)

The Streamlit app's successor: a React + Mantine frontend here, a FastAPI service in
`app/api/`, both over the same `app/logic/` and `app/data/`. Only 🚀 Momentum Analysis is
ported so far; the Streamlit app keeps running until every page is.

```
Firebase Hosting ── web/dist (this app)
   └─ /api/** ──► Cloud Run "equiti-api" (deploy/api/Dockerfile → uvicorn app.api.main:app)
```

## Run it locally

Two processes. From the repository root:

```bash
pip install -r requirements-dev.txt
```

```bash
EQUITI_AUTH=off python3 -m uvicorn app.api.main:app --port 8600 --reload --reload-dir app
```

```bash
cd web && npm install && npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` to port 8600 (`API_PORT` overrides it), the
same shape Hosting gives production. The API reads `SHARADAR_API_KEY` from the environment or
from `.streamlit/secrets.toml`. API docs are at <http://localhost:8600/api/docs>.

`EQUITI_AUTH=off` skips sign-in; it is ignored on Cloud Run. To try sign-in locally, set the
`VITE_FIREBASE_*` values in `web/.env.local` (see `.env.example`) and run the API with
`FIREBASE_PROJECT_ID` and `EQUITI_ALLOWED_EMAILS` instead.

## After changing an API response

```bash
cd web && npm run gen:api
```

That regenerates `src/api/schema.ts` from FastAPI's OpenAPI schema. A renamed or retyped field
on the Python side then fails `npm run typecheck` rather than showing up as `undefined`.

## Rules the code relies on

- **Missing numbers are `null`, on purpose.** Every response model derives from
  `app.api.contract.Contract`, which rejects NaN and infinity. A value that can be absent is typed
  `float | None` and converted with `finite_or_none`; a stray NaN fails the request rather than
  being silently renamed `null` (which is what Pydantic otherwise does, even for a plain `float`).
- **No result is drawn against inputs it does not match.** The divergence query keeps its
  previous answer while the next one loads (`placeholderData`), but shows it dimmed and keeps it
  off the chart. The DCF tab will need the same rule for its assumption fingerprint.
- **Every data route requires sign-in** (`test_every_data_route_carries_the_gate`). The data is
  licensed, and Hosting can only reach a Cloud Run service open to unauthenticated calls, so the
  gate is a Firebase ID token plus an email allowlist. An empty allowlist admits nobody.

## Deploy

One-time setup, in the GCP project the options collector already uses (`PROJECT`, `REGION`
as in `collector/README.md`):

1. **Firebase**: in the Firebase console, add Firebase to the existing Google Cloud project.
   Under Authentication, enable the Google provider. Register a Web app and copy its config
   into `web/.env.production.local` (`VITE_FIREBASE_*`).
2. **Secret and service account**:

   ```bash
   printf '%s' 'YOUR_SHARADAR_KEY' | gcloud secrets create sharadar-api-key --data-file=-
   gcloud iam service-accounts create equiti-api
   gcloud secrets add-iam-policy-binding sharadar-api-key --member=serviceAccount:equiti-api@PROJECT.iam.gserviceaccount.com --role=roles/secretmanager.secretAccessor
   ```

Each release:

```bash
gcloud builds submit --config deploy/api/cloudbuild.yaml --substitutions=_REGION=REGION --service-account=projects/PROJECT/serviceAccounts/options-builder@PROJECT.iam.gserviceaccount.com .
```

```bash
gcloud run deploy equiti-api --image REGION-docker.pkg.dev/PROJECT/options/api:latest --region REGION --service-account equiti-api@PROJECT.iam.gserviceaccount.com --allow-unauthenticated --set-secrets SHARADAR_API_KEY=sharadar-api-key:latest --set-env-vars "^;^FIREBASE_PROJECT_ID=PROJECT;EQUITI_ALLOWED_EMAILS=you@example.com" --memory 1Gi --timeout 60 --max-instances 2
```

```bash
cd web && npm run build && npx firebase-tools deploy --only hosting --project PROJECT
```

Notes:

- `--allow-unauthenticated` is required because Hosting's rewrite carries no IAM identity. The app's
  own token check is the gate.
- The `^;^` prefix makes `;` the separator, so the email list can keep its commas.
- `firebase.json` assumes `us-central1`; change its `region` if the service is elsewhere.
- Hosting stops waiting for Cloud Run after **60 seconds**. A cold Sharadar price fetch takes 5–20 s.
- The data layer's caches are still `st.cache_data`, in process: each instance starts cold, and
  `--max-instances 2` keeps that bounded. Moving them to a shared cache is the next step of the
  migration.
