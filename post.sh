echo $1
curl -d '{"dataframe_split": {"columns": ["text"], "data": [["$1"]]}}' -H 'Content-Type: application/json' -X POST localhost:5002/invocations