import numpy as np
import sys
# emb_mt_name = 'opus-mt-de-en/embedding'
emb_mt_name = sys.argv[1]

emb_list = []
for line in open(emb_mt_name):
    line = line.strip().split()
    if len(line) == 2:
        continue
    token = line[0]
    emb = line[1:]
    emb = [float(x) for x in emb]
    assert len(emb) == 512
    emb_list.append(emb)
emb_list = np.array(emb_list)
cov = np.cov(emb_list.T)

mean = np.mean(emb_list)
std = np.std(emb_list)
print(mean, std)
# 0.0013373970354963827 0.04711890907013472
mean = np.mean(emb_list, axis=0)
# a = np.random.multivariate_normal(mean, cov, size=1).tolist()[0]