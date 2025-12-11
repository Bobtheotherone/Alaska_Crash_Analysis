# Deployment Guide

This document describes how to run the Alaska Crash Analysis stack in local
development and in a simple containerized (Docker) setup.

## 1. Environment variables

The Django settings module reads all sensitive values from the environment.

Required in non-DEBUG environments:

- `DJANGO_SECRET_KEY`
- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_HOST`
- `POSTGRES_PORT`

Recommended (but optional) variables:

- `DJANGO_DEBUG`
- `DJANGO_ALLOWED_HOSTS`
- `CORS_ALLOW_ALL_ORIGINS`
- `DJANGO_SECURE_SSL_REDIRECT`
- `DJANGO_SECURE_HSTS_SECONDS`
- `DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS`
- `DJANGO_SECURE_HSTS_PRELOAD`
- `INGESTION_MAX_FILE_SIZE_BYTES`
- `INGESTION_ALLOWED_EXTENSIONS`
- `INGESTION_SCHEMA_CONFIG_PATH`
- `INGESTION_REQUIRE_AV`
- `CLAMAV_UNIX_SOCKET` or `CLAMAV_TCP_HOST` / `CLAMAV_TCP_PORT`

For ingestion, a typical production configuration is:

```bash
INGESTION_ALLOWED_EXTENSIONS=".csv,.parquet"
INGESTION_MAX_FILE_SIZE_BYTES=10485760  # 10 MB
```

Adjust these as needed for your environment (e.g. to restrict uploads to CSV
only, set `INGESTION_ALLOWED_EXTENSIONS=".csv"`).

## 2. Local development (without Docker)

1. Start Postgres/PostGIS locally and create a database:

   ```bash
   createdb alaska_crash_analysis
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure environment variables (e.g. in a `.env` file or by exporting
   them in your shell).

   At minimum, you likely want:

   ```bash
   export DJANGO_DEBUG=true
   export POSTGRES_DB=alaska_crash_analysis
   export POSTGRES_USER=postgres
   export POSTGRES_PASSWORD=postgres
   export POSTGRES_HOST=localhost
   export POSTGRES_PORT=5432
   # Optional: ingestion tuning
   export INGESTION_ALLOWED_EXTENSIONS=".csv,.parquet"
   export INGESTION_MAX_FILE_SIZE_BYTES=10485760
   ```

4. Run migrations and start the dev server:

   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

5. In another terminal, start the frontend dev server:

   ```bash
   cd alaska_ui
   npm install
   npm run dev
   ```

## 3. Docker / docker-compose

A minimal `docker-compose.yml` is provided with three services:

- `db` - Postgres + PostGIS
- `clamav` - ClamAV daemon for antivirus scanning
- `web` - Django

Bring up the stack with:

```bash
docker compose up --build
```

This starts PostGIS (`db`), a ClamAV daemon on TCP 3310 (`clamav`, reachable from Django as `clamav:3310`), and the Django backend (`web`). In this containerized setup, clean uploads should report `AV_SCAN` with `status="passed"` and `severity="info"` because ClamAV is available by default.

`.env` is optional for this stack; `docker compose up --build` works without it, but you can supply one to override environment variables if needed.

### Antivirus in non-Docker environments

- Install ClamAV/`clamd` on Linux or WSL (e.g., `apt install clamav clamav-daemon`) and start the daemon.
- Configure either `CLAMAV_UNIX_SOCKET` **or** `CLAMAV_TCP_HOST` / `CLAMAV_TCP_PORT` before running `python manage.py runserver`.
- If ClamAV is not installed and `INGESTION_REQUIRE_AV=false` (default), uploads are allowed but the `AV_SCAN` step is recorded as `skipped` with `severity="warning"`.
- If `INGESTION_REQUIRE_AV=true` and ClamAV is unavailable, uploads are rejected with `error_code="AV_UNAVAILABLE_REQUIRED"`.
