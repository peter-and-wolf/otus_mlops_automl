# otus_mlops_automl

## Pycaret в Spark-кластере

Запускаем кластер из Docker-контейнеров:

```bash
cd spark
docker-compose up -d
```

Запускаем Jupyter Notebook внутри кластера:

```bash
bash run_spark_notebook.sh
```