# Deploys

A push to `main` (a merged pull request) deploys whatever it changed, once the tests pass —
`.github/workflows/deploy.yml`:

| Changed | Deployed |
|---|---|
| `app/`, `requirements.txt`, `deploy/api/` | the API, Cloud Run service `equiti-api` |
| `collector/`, and the app modules its image copies | the collector, Cloud Run job `options-collector` |
| `web/`, `firebase.json` | the web app, Firebase Hosting |

A change that touches none of them (docs, tests) deploys nothing. The web app goes last, so a
page never reaches readers before the API it calls. Each image is tagged with its commit, and the
API is checked once it is serving: `/api/health` answers, and a data route still refuses a caller
who is not signed in. The collector is only pointed at its new image; its next scheduled run uses
it.

**Actions → deploy → Run workflow** deploys by hand, naming any of `api collector web`. The
commands in `web/README.md` and `collector/README.md` still work as a fallback.

To roll back, revert the change on `main`; the revert deploys like any other merge. In a hurry,
the Cloud Run console's *Revisions* tab and the Hosting console's release history can each switch
back to the previous release in place — but a traffic split pinned to an old revision stays
pinned, so send traffic back to the latest revision once the revert has deployed.

## One-time setup

Replace `PROJECT`, `PROJECT_NUMBER` and `REGION` as in `collector/README.md`, `OWNER/REPO` with
the GitHub repository, and `REPO_ID` with its numeric id (`gh api repos/OWNER/REPO --jq .id`).

### 1. Let GitHub's workflow sign in, without a key

Workload Identity Federation exchanges the workflow's own GitHub token for a short-lived Google
one, so no service-account key exists to leak. The provider trusts only `deploy.yml`, running on
`main`, in this repository — named by its id, so a re-created repository of the same name
inherits nothing — and a pull request, which runs on its own ref, can never deploy.

```bash
gcloud services enable sts.googleapis.com iamcredentials.googleapis.com --project PROJECT
gcloud iam workload-identity-pools create github --project PROJECT --location global --display-name "GitHub Actions"
gcloud iam workload-identity-pools providers create-oidc deploy --project PROJECT --location global \
  --workload-identity-pool github --display-name "deploy.yml on main" \
  --issuer-uri https://token.actions.githubusercontent.com \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository_id=assertion.repository_id" \
  --attribute-condition "assertion.repository_id == 'REPO_ID' && assertion.ref == 'refs/heads/main' && assertion.workflow_ref == 'OWNER/REPO/.github/workflows/deploy.yml@refs/heads/main'"
```

### 2. The account the workflow deploys as

It can push images to the one repository, deploy Cloud Run services and jobs, run them as exactly
the two accounts they already run as, and release to Hosting. It cannot change IAM, read the
Sharadar key or touch the bucket. The Firebase CLI bills its API calls to the project, hence
`serviceUsageConsumer`.

```bash
gcloud iam service-accounts create github-deployer --project PROJECT --display-name "GitHub Actions deploys"
D=github-deployer@PROJECT.iam.gserviceaccount.com
gcloud iam service-accounts add-iam-policy-binding $D --project PROJECT --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository_id/REPO_ID"
gcloud artifacts repositories add-iam-policy-binding options --project PROJECT --location REGION \
  --member serviceAccount:$D --role roles/artifactregistry.writer
for role in roles/run.developer roles/firebasehosting.admin roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding PROJECT --member serviceAccount:$D --role $role
done
for runtime in equiti-api options-collector; do
  gcloud iam service-accounts add-iam-policy-binding $runtime@PROJECT.iam.gserviceaccount.com --project PROJECT \
    --member serviceAccount:$D --role roles/iam.serviceAccountUser
done
```

### 3. Tell the workflow where to deploy

These are repository **secrets** although none is a credential: the repository is public, so
its Actions logs are too, and a secret is masked wherever it would print. The project number
is masked as well, though it is no secret of its own: each deploy job takes it out of
`GCP_WORKLOAD_IDENTITY_PROVIDER` before signing in, because `gcloud run deploy` prints it in
the service's URL. The four
`VITE_FIREBASE_*` values are the web app's Firebase config, read from the gitignored
`web/.env.production.local`.

```bash
gh secret set GCP_PROJECT_ID --body PROJECT
gh secret set GCP_DEPLOYER --body github-deployer@PROJECT.iam.gserviceaccount.com
gh secret set GCP_WORKLOAD_IDENTITY_PROVIDER --body projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/deploy
gh secret set -f web/.env.production.local
```

### 4. Retire the collector's Cloud Build trigger

If the project still has the `options-builder` trigger from an earlier setup, delete it: it built
the collector's image on every push to `main` and deployed none of them.

```bash
gcloud builds triggers delete options-builder --project PROJECT
```
