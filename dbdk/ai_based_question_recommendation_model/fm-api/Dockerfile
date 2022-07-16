FROM python:3.7
LABEL maintainer="kay.seo@storypot.io"

ENV PYTHONUNBUFFERED 1

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime

RUN apt-get update
RUN apt-get upgrade -y

RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/ubuntu/18.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN apt-get update

RUN ACCEPT_EULA=Y apt-get install -y msodbcsql17
RUN apt-get install -y gcc binutils cmake git unixodbc-dev
RUN apt-get clean && apt-get autoremove -y

RUN chmod +rwx /etc/ssl/openssl.cnf
RUN sed -i 's/TLSv1.2/TLSv1/g' /etc/ssl/openssl.cnf
RUN sed -i 's/SECLEVEL=2/SECLEVEL=1/g' /etc/ssl/openssl.cnf

WORKDIR /data/fm/source
RUN git clone https://github.com/aksnzhy/xlearn.git

WORKDIR /data/fm/source/xlearn/build
RUN cmake ../ && make

RUN python -m pip install --upgrade pip setuptools wheel
ADD requirements.txt /data/fm/
RUN pip install -r /data/fm/requirements.txt

WORKDIR /data/fm/source/xlearn/build/python-package
RUN ./install-python.sh

ADD ./config/ssh_key/id_rsa /root/.ssh/
ADD ./config/ssh_key/id_rsa.pub /root/.ssh/
ADD ./config/ssh_key/known_hosts /root/.ssh/
RUN chmod 600 /root/.ssh/id_rsa
RUN chmod 644 /root/.ssh/id_rsa.pub
RUN chmod 644 /root/.ssh/known_hosts
RUN ssh-keygen -R 183.110.210.105

RUN git clone ssh://git@183.110.210.105:2224/root/fm-model.git /data/fm/persistent

ADD . /data/fm/fm-api

WORKDIR /data/fm/fm-api
