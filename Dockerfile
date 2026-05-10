FROM tensorflow/tensorflow:2.15.0-gpu

LABEL maintainer="yunusemremidilli@gmail.com"

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

COPY ./requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

ENV TF_ENABLE_ONEDNN_OPTS=0

# TODO: Disable below code in production
# COPY /task /app
WORKDIR /app

# ENTRYPOINT [ "python" ]
