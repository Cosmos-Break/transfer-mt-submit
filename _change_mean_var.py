import numpy as np
import sys
from sklearn import preprocessing

emb_mt_name = 'opus-mt-de-en/embedding'
# emb_mt_name_lang = sys.argv[1]
emb_mt_name_lang = 'eu-fasttext-model.vec_underline'
emb_mt_name_lang_norm = f'{emb_mt_name_lang}_norm'

def get_emb_list(emb_name):
    emb_list = []
    for line in open(emb_name):
        line = line.strip().split()
        if len(line) == 2:
            continue
        token = line[0]
        emb = line[1:]
        emb = [float(x) for x in emb]
        assert len(emb) == 512
        emb_list.append(emb)
    emb_list = np.array(emb_list)
    return emb_list
emb_list = get_emb_list(emb_mt_name)
emb_list_lang = get_emb_list(emb_mt_name_lang)
# cov = np.cov(emb_list.T)
mean = np.mean(emb_list)
std = np.std(emb_list)
print(mean, std)
# 0.0013373970354963827 0.04711890907013472


mean = np.mean(emb_list_lang, axis=0)
std = np.std(emb_list_lang, axis=0)
var = std * std
print('mean: {}, std: {}, var: {}'.format(mean, std, var))
# numpy 的广播功能
another_trans_data = emb_list_lang - mean
# 注：是除以标准差
another_trans_data = another_trans_data / std
print('another_trans_data: ')
print(another_trans_data)

print(np.mean(another_trans_data))
print(np.std(another_trans_data))


mean = np.mean(emb_list, axis=0)
std = np.std(emb_list, axis=0)
another_trans_data = another_trans_data * std
another_trans_data = emb_list_lang + mean
print(np.mean(another_trans_data))
print(np.std(another_trans_data))

with open(emb_mt_name_lang_norm, 'w') as f:
    fin = open(emb_mt_name_lang)
    line = fin.readline()
    f.write(line)
    another_trans_data = another_trans_data.tolist()
    for line, emb in zip(fin, another_trans_data):
        line = line.strip().split()
        token = line[0]
        emb = [str(round(x, 7)) for x in emb]
        assert len(emb) == 512
        f.write(' '.join([token] + emb) + '\n')


# scaler = preprocessing.StandardScaler(mean)
print(123)