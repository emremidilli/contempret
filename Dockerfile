FROM tensorflow/tensorflow:2.13.0rc2-gpu

LABEL maintainer="yunusemremidilli@gmail.com"

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

COPY ./requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

COPY /task /app
WORKDIR /app

# ENTRYPOINT [ "python" ]
