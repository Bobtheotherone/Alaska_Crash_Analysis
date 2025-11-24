from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Core Django settings
# ---------------------------------------------------------------------------

DEBUG = os.environ.get("DJANGO_DEBUG", "true").lower() == "true"

# SECRET_KEY must always come from the environment when DEBUG is False.
if DEBUG:
    SECRET_KEY = os.environ.get(
        "DJANGO_SECRET_KEY",
        "dev-secret-key-not-for-production",
    )
else:
    SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]

# More secure default: only localhost unless DJANGO_ALLOWED_HOSTS is set.
# Example: DJANGO_ALLOWED_HOSTS="example.com,api.example.com"
ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get(
        "DJANGO_ALLOWED_HOSTS",
        "127.0.0.1,localhost",
    ).split(",")
    if host.strip()
]

# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.gis",
    "rest_framework",
    "django_filters",
    "drf_spectacular",
    "corsheaders",
    "ingestion",
    "crashdata",
    "analysis",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "alaska_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "alaska_project.wsgi.application"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DATABASES = {
    "default": {
        "ENGINE": "django.contrib.gis.db.backends.postgis",
        "NAME": os.environ.get("POSTGRES_DB", "alaska_crash_analysis"),
        "USER": os.environ.get("POSTGRES_USER", "postgres"),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD", ""),
        "HOST": os.environ.get("POSTGRES_HOST", "localhost"),
        "PORT": os.environ.get("POSTGRES_PORT", "5432"),
    }
}

# ---------------------------------------------------------------------------
# Password validation
# ---------------------------------------------------------------------------

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# ---------------------------------------------------------------------------
# Internationalization
# ---------------------------------------------------------------------------

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

STATIC_URL = "/static/"
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")

MEDIA_ROOT = BASE_DIR / "media"
MEDIA_URL = "/media/"


DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# REST framework configuration
# ---------------------------------------------------------------------------

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_FILTER_BACKENDS": [
        "django_filters.rest_framework.DjangoFilterBackend",
        "rest_framework.filters.OrderingFilter",
        "rest_framework.filters.SearchFilter",
    ],
    "DEFAULT_THROTTLE_RATES": {
        # Upload gateway: fairly low by default to avoid abuse.
        "ingest_upload": "10/hour",
        # Export endpoints: lower rate to avoid accidental DoS.
        "exports": "5/hour",
    },
}

SPECTACULAR_SETTINGS = {
    "TITLE": "Alaska Crash Analysis API",
    "DESCRIPTION": "Django backend for ingesting and exploring Alaska crash data.",
    "VERSION": "1.0.0",
}

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

CORS_ALLOW_ALL_ORIGINS = os.environ.get("CORS_ALLOW_ALL_ORIGINS", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Upload gateway / ingestion configuration
# ---------------------------------------------------------------------------

INGESTION_MAX_FILE_SIZE_BYTES = int(
    os.environ.get("INGESTION_MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024))  # 10 MB
)

# Comma-separated list of allowed file extensions for uploads.
# Production deployments should set this via environment or .env.
# Example: INGESTION_ALLOWED_EXTENSIONS=".csv,.parquet"
INGESTION_ALLOWED_EXTENSIONS = [
    ext.strip().lower()
    for ext in os.environ.get(
        "INGESTION_ALLOWED_EXTENSIONS",
        ".csv,.parquet",
    ).split(",")
    if ext.strip()
]

# When True, antivirus scanning is *required* for ingestion. If ClamAV is
# unavailable and this flag is True, uploads will be rejected.
INGESTION_REQUIRE_AV = os.environ.get("INGESTION_REQUIRE_AV", "false").lower() == "true"

# Path to the externalized MMUCC/KABCO schema configuration that non-devs can
# edit without touching Python code.
INGESTION_SCHEMA_CONFIG_PATH = os.environ.get(
    "INGESTION_SCHEMA_CONFIG_PATH",
    str(BASE_DIR / "ingestion" / "config" / "mmucc_schema.yml"),
)

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
SECURE_SSL_REDIRECT = os.environ.get("DJANGO_SECURE_SSL_REDIRECT", "false").lower() == "true"

# You may wish to fine-tune this further for production deployments.
SECURE_HSTS_SECONDS = int(os.environ.get("DJANGO_SECURE_HSTS_SECONDS", "0"))
SECURE_HSTS_INCLUDE_SUBDOMAINS = os.environ.get(
    "DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS", "false"
).lower() == "true"
SECURE_HSTS_PRELOAD = os.environ.get("DJANGO_SECURE_HSTS_PRELOAD", "false").lower() == "true"
