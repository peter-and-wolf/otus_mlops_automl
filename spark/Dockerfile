FROM python:3.11-bullseye AS spark-base

ARG SPARK_VERSION=3.5.2

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      sudo \
      curl \
      vim \
      unzip \
      rsync \
      screen \
      openjdk-11-jdk \
      build-essential \
      software-properties-common \
      ssh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Optional env variables
ENV SPARK_HOME="/opt/spark"
ENV HADOOP_HOME="/opt/hadoop"

RUN mkdir -p ${HADOOP_HOME} && mkdir -p ${SPARK_HOME}
WORKDIR ${SPARK_HOME}

# Download and install Spark
RUN curl https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz -o spark-${SPARK_VERSION}-bin-hadoop3.tgz \
  && tar xvzf spark-${SPARK_VERSION}-bin-hadoop3.tgz --directory /opt/spark --strip-components 1 \
  && rm -rf spark-${SPARK_VERSION}-bin-hadoop3.tgz

FROM spark-base AS pyspark

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ENV SPARK_MASTER="spark://spark-master:7077"
ENV SPARK_MASTER_PORT=7077
ENV SPARK_MASTER_WEBUI_PORT=8080
ENV SPARK_WORKER_PORT=7000
ENV SPARK_WORKER_WEBUI_PORT=8080
ENV SPARK_NO_DAEMONIZE=true
ENV PATH="/opt/spark/sbin:/opt/spark/bin:$PATH"
ENV PYSPARK_PYTHON=python3

COPY conf/spark-defaults.conf "$SPARK_HOME/conf"

RUN chmod u+x /opt/spark/sbin/* && \
    chmod u+x /opt/spark/bin/*

ENV PYTHONPATH="$SPARK_HOME/python/:$PYTHONPATH"

COPY start_spark.sh .
