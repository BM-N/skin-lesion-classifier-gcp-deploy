FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install dependencies
COPY ./project/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# source code
COPY ./project /app/project
COPY ./project/pyproject.toml /app/pyproject.toml

# install the project in editable mode
RUN pip install -e /app/project

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "project.api.main:app"]