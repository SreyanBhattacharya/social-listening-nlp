FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Train inside the image so the container is self-contained (downloads the NLTK corpus).
RUN python -m sociallistening.train --reports-dir /tmp/reports

EXPOSE 8000
CMD ["uvicorn", "sociallistening.api:app", "--host", "0.0.0.0", "--port", "8000"]
