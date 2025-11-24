FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for Postgres + GeoDjango
RUN apt-get update && apt-get install -y --no-install-recommends         build-essential         libpq-dev         binutils         libproj-dev         gdal-bin         && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Collect static files if the frontend has been built into the expected path.
RUN python manage.py collectstatic --noinput || true

CMD ["gunicorn", "alaska_project.wsgi:application", "--bind", "0.0.0.0:8000"]
