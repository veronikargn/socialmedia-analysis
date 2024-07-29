# Use the latest Apache Airflow image as base
FROM apache/airflow:latest

# Switch to the root user to install system packages
USER root

# Install system packages
RUN apt-get update && \
    apt-get -y install git && \
    apt-get -y install python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Switch back to the airflow user
USER airflow

# Upgrade Pydantic
RUN pip install --no-cache-dir --upgrade pydantic

# Install Python packages required for the Airflow tasks
RUN pip install --no-cache-dir \
    pendulum \
    pymongo \
    pandas \
    praw \
    gensim \
    datetime \
    numpy \
    scikit-learn \
    nltk \
    scipy==1.12 \
    joblib

# NLTK: Download stopwords corpus
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet