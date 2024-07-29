# Dockerfile for Python Scripts
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# NLTK: Download stopwords corpus
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet

COPY . .

CMD ["bash", "start.sh"]
