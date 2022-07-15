FROM python:3.7
LABEL maintainer="kay.seo@storypot.io"

ENV PYTHONUNBUFFERED 1

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get clean && apt-get autoremove -y

RUN python -m pip install --upgrade pip setuptools wheel

ADD requirements.txt /data/dkt/
RUN pip install -r /data/dkt/requirements.txt

WORKDIR /data/dkt/dkt-batch
