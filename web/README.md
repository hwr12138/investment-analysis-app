# Equiti web (proof of concept)

The Streamlit app's successor: a React + Mantine frontend here, a FastAPI service in
`app/api/`, both over the same `app/logic/` and `app/data/`. 🚀 Momentum Analysis, 🔎 Stock Research
(all six tabs, the 💰 DCF included), the two portfolio pages and 🎯 Options are ported so far; the
Streamlit app keeps running until every page is.

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
  off the chart. The DCF tab does the same with its overrides as the query key: while the next run
  loads, the last one stays on screen dimmed, headline included, and a refused input (a 422) leaves
  it dimmed beside an Undo. Only the inputs the reader changed are sent, so an untouched one keeps
  following its default.
- **Every data route requires sign-in** (`test_every_data_route_carries_the_gate`). The data is
  licensed, and Hosting can only reach a Cloud Run service open to unauthenticated calls, so the
  gate is a Firebase ID token plus an email allowlist. An empty allowlist admits nobody.

## Deploy

It lives in the same Google Cloud project as the options collector (`PROJECT`, `REGION` as in
`collector/README.md`), served at **<https://equiti-app.web.app>**. That is a second Hosting site
named in `firebase.json`; the project's default site would carry the permanent project ID instead.

One-time setup:

1. **Firebase**:

   ```bash
   firebase projects:addfirebase PROJECT
   firebase hosting:sites:create equiti-app --project PROJECT
   firebase apps:create WEB Equiti --project PROJECT
   ```

   Copy `firebase apps:sdkconfig WEB <app id> --project PROJECT` into
   `web/.env.production.local` as the four `VITE_FIREBASE_*` values (git ignores the file). Then,
   in the Firebase console under Authentication → Sign-in method, turn on **Google**. The CLI
   cannot do that step, because it creates the OAuth client. Finally add `equiti-app.web.app` and
   `equiti-app.firebaseapp.com` under Authentication → Settings → Authorized domains. Firebase
   lists only the project's default domains, so sign-in on the second site fails until they are
   added.
2. **Secret and service account** (the build account also needs the staging-bucket binding in
   `collector/README.md` §4):

   ```bash
   printf '%s' 'YOUR_SHARADAR_KEY' | gcloud secrets create sharadar-api-key --data-file=-
   gcloud iam service-accounts create equiti-api
   gcloud secrets add-iam-policy-binding sharadar-api-key --member=serviceAccount:equiti-api@PROJECT.iam.gserviceaccount.com --role=roles/secretmanager.secretAccessor
   ```

3. **Read access to the options store**, for the 🎯 Options routes. The service authenticates to
   the bucket as its own account, so it needs no key file:

   ```bash
   gcloud storage buckets add-iam-policy-binding gs://BUCKET --member=serviceAccount:equiti-api@PROJECT.iam.gserviceaccount.com --role=roles/storage.objectViewer
   ```

4. **A custom domain**, optional, here a subdomain on a Cloud DNS zone (`ZONE`). Register it on
   the site (the console's Hosting → `equiti-app` → *Add custom domain* does the same); Firebase
   answers with the one record it wants, a CNAME to `equiti-app.web.app`:

   ```bash
   curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "x-goog-user-project: PROJECT" -H "Content-Type: application/json" -d '{}' "https://firebasehosting.googleapis.com/v1beta1/projects/PROJECT/sites/equiti-app/customDomains?customDomainId=equiti.DOMAIN"
   ```

   ```bash
   gcloud dns record-sets create equiti.DOMAIN. --zone ZONE --type CNAME --ttl 300 --rrdatas equiti-app.web.app.
   ```

   Then add `equiti.DOMAIN` to Authentication → Settings → Authorized domains, as in step 1, or
   the Google popup refuses to open from it. The certificate follows on its own once the CNAME
   resolves. Nothing else changes: the page calls `/api` on whichever domain served it, and the
   API checks the token's project, not the domain it came from.

Each release:

```bash
gcloud builds submit --config deploy/api/cloudbuild.yaml --substitutions=_REGION=REGION --service-account=projects/PROJECT/serviceAccounts/options-builder@PROJECT.iam.gserviceaccount.com .
```

```bash
gcloud run deploy equiti-api --image REGION-docker.pkg.dev/PROJECT/options/api:latest --region REGION --service-account equiti-api@PROJECT.iam.gserviceaccount.com --allow-unauthenticated --set-secrets SHARADAR_API_KEY=sharadar-api-key:latest --set-env-vars "^;^FIREBASE_PROJECT_ID=PROJECT;EQUITI_ALLOWED_EMAILS=you@example.com;OPTIONS_STORE_URI=gs://BUCKET" --memory 1Gi --timeout 60 --max-instances 2 --cpu-boost
```

```bash
cd web && npm run build && npx firebase-tools deploy --only hosting --project PROJECT
```

Notes:

- `--allow-unauthenticated` is required because Hosting's rewrite carries no IAM identity. The app's
  own token check is the gate.
- The `^;^` prefix makes `;` the separator, so the email list can keep its commas.
- `firebase.json` assumes `us-central1`; change its `region` if the service is elsewhere.
- Every page is served `no-cache` and only the hashed `assets/` are cached for a year, so a
  release shows on the next load. A rule on `/index.html` alone would miss routes such as
  `/momentum/AAPL`, which reach it through the rewrite and otherwise get Hosting's one-hour cache.
- Hosting stops waiting for Cloud Run after **60 seconds**. A cold Sharadar price fetch takes 5–20 s.
- **The Options routes solve nothing heavy on request.** Solving every name's surface takes the
  Streamlit page about ten seconds, so the collector does it nightly and stores the Discover and
  Screen tables (~130 kB); the API reads those in milliseconds. Evaluate reads the night's full
  chain once per instance (~35 MB in memory) and solves one name, about 0.3 s. A store whose
  tables are older than its manifest's close answers 503 with how to rebuild them
  (`collector/README.md`, "Rebuilding the screens").
- The data layer's caches are still `st.cache_data`, in process: each instance starts cold, and
  `--max-instances 2` keeps that bounded. Moving them to a shared cache is the next step of the
  migration.
- **The first request after an idle spell waits 12–15 s** while Cloud Run starts an instance,
  because there is no minimum instance. Locally the app imports in about 1 s, so the time is
  mostly Cloud Run loading the ~300 MB image. `--cpu-boost` made no measurable difference (14.4 s
  with it, 11.6–15.7 s without). `--min-instances 1` would remove the wait at about $13 a month
  idle for 1 vCPU and 1 GiB; the wait was accepted instead (November 2026). A smaller image is the
  other way out: the API currently installs everything the Streamlit app needs.
