from urllib import request
import random, os
import numpy as np
import torch
from datasets import load_metric

metric_f1 = load_metric("f1")
metric_precision = load_metric("precision")
metric_recall = load_metric("recall")

def pull_data_manager():
    module_url = f"https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/semeval-2022/dont_patronize_me.py"
    module_name = module_url.split('/')[-1]
    print(f'Fetching {module_url}')
    #with open("file_1.txt") as f1, open("file_2.txt") as f2
    with request.urlopen(module_url) as f, open(module_name,'w') as outf:
        a = f.read()
        outf.write(a.decode('utf-8'))
    
def labels2file(p, outf_path):
	with open(outf_path,'w') as outf:
		for pi in p:
			outf.write(','.join([str(k) for k in pi])+'\n')

def file2labels(path):
    with open(path) as f:
        return [int(line) for line in f] 

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True