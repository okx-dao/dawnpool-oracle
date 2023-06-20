FROM python:3.9.14-slim-bullseye as base

ENV LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN sed -i "s@http://\(deb\|security\).debian.org@https://mirrors.aliyun.com@g" /etc/apt/sources.list
RUN apt-get update \
    && apt-get install -y vim wget net-tools curl git gcc libffi-dev g++ \
    && curl --version \
    && git --version \
    && apt-get clean
#    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM base as builder

ENV POETRY_VERSION=1.4.2 \
    POETRY_HOME="/opt/poetry"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN cd /home && wget https://github.com/python-poetry/poetry/releases/download/1.4.2/poetry-1.4.2-py3-none-any.whl \
    && /usr/local/bin/python -m pip install --upgrade pip && pip install wheel && pip --default-timeout=6000 install -i https://mirrors.aliyun.com/pypi/simple ./poetry-1.4.2-py3-none-any.whl
ENV PATH="$POETRY_HOME/bin:$PATH"

COPY pyproject.toml poetry.lock ./
RUN python -m venv --copies /venv

RUN . /venv/bin/activate && poetry install

FROM base as production

COPY --from=builder /venv /venv

RUN mkdir -p /var/www && chown www-data /var/www && \
    chown -R www-data /app/ && chown -R www-data /venv

ENV PYTHONPATH="/venv/lib/python3.9/site-packages/"
ENV PATH=$PATH:/venv/bin
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PULSE_SERVER_PORT 8000
ENV PROMETHEUS_PORT 8000

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
ENV VERSION=${VERSION}
ENV COMMIT_DATETIME=${COMMIT_DATETIME}
ENV BUILD_DATETIME=${BUILD_DATETIME}
ENV TAGS=${TAGS}
ENV BRANCH=${BRANCH}
ENV COMMIT_MESSAGE=${COMMIT_MESSAGE}
ENV COMMIT_HASH=${COMMIT_HASH}

EXPOSE $PROMETHEUS_PORT
USER www-data

COPY --from=builder /usr/local/ /usr/local/
COPY assets ./assets
COPY app ./

HEALTHCHECK --interval=10s --timeout=3s \
    CMD curl -f http://localhost:$PULSE_SERVER_PORT/healthcheck || exit 1

ENTRYPOINT ["python3", "-u", "oracle.py"]