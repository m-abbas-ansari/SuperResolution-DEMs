import yaml
from data import DemDataset

opt = yaml.load(open('test.yaml'), Loader=yaml.SafeLoader)
print(opt.generator)