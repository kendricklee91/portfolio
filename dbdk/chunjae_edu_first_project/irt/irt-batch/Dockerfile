FROM python:3.6
LABEL maintainer="kay.seo@storypot.io"

ENV PYTHONUNBUFFERED 1

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get clean && apt-get autoremove -y

RUN python -m pip install --upgrade pip setuptools wheel

ADD requirements.txt /data/irt/
ADD wheel/tensorflow-1.6.0-cp36-cp36m-linux_x86_64.whl /data/irt/packages/

RUN pip install -r /data/irt/requirements.txt

WORKDIR /data/irt/irt-batch
