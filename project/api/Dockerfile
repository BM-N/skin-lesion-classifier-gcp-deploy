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

CMD ["uvicorn", "project.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
