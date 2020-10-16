import yaml
import os
from os import getcwd


print(getcwd())
curPath = os.path.join("projects/shoe.yml")
f = open(curPath, 'r', encoding='utf-8')
cfg = f.read()
d = yaml.load(cfg)
print(d['train_txt'])


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

opt = 'shoe'
params = Params(f'projects/{opt}.yml')
print(params.train_txt)