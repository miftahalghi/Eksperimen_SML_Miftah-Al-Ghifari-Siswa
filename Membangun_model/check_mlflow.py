import requests, json
r = requests.get('http://localhost:5000/api/2.0/mlflow/experiments/search', params={'max_results': 10})
data = r.json()
exps = data.get('experiments', [])
print(f"Total experiments: {len(exps)}")
for e in exps:
    print(f"  ID={e['experiment_id']}, Name={e['name']}")
