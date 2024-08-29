FROM python:3.11-bullseye 

ENV DASK_HOME="/opt/dask"
WORKDIR ${DASK_HOME}

COPY requirements.txt .
RUN pip3 install -r requirements.txt