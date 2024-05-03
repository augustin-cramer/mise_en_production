FROM python:3.10-slim

ARG ENDPOINT_URL
ARG S3_KEY
ARG S3_SECRET
ARG S3_TOKEN

ENV ENDPOINT_URL=${ENDPOINT_URL}
ENV S3_KEY=${S3_KEY}
ENV S3_SECRET=${S3_SECRET}
ENV S3_TOKEN=${S3_TOKEN}

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]