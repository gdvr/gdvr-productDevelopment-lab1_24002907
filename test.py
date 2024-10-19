import yaml
with open('params.yaml') as f:
    params = yaml.safe_load(f)
print(params)