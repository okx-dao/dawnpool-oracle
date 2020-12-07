FROM python:3.8-slim

# Build

RUN apt-get update \
 && apt-get install -y gcc \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY assets ./assets
COPY app ./

ENTRYPOINT ["python3", "-u", "oracle.py"]

# Set metadata

ARG VERSION
ARG COMMIT_DATETIME
ARG BUILD_DATETIME
ARG TAGS
ARG BRANCH
ARG COMMIT_MESSAGE
ARG COMMIT_HASH
LABEL VERSION="$VERSION"
LABEL COMMIT_DATETIME="$COMMIT_DATETIME"
LABEL BUILD_DATETIME="$BUILD_DATETIME"
LABEL TAGS="$TAGS"
LABEL BRANCH="$BRANCH"
LABEL COMMIT_MESSAGE="$COMMIT_MESSAGE"
LABEL COMMIT_HASH="$COMMIT_HASH"
COPY ./version.json /version.json
