from collections import defaultdict
import numpy as np
from numpy.random import random
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTokenizedInput
from tokenizers import CharBPETokenizer
from tokenizers.pre_tokenizers import Whitespace
from transformers.utils.dummy_pt_objects import SqueezeBertModel
import numpy as np
np.random.seed(42)
dic = {}
dic_avg = defaultdict(list)
for line in open('en-my.dic.all'):
    line = line.strip().split()
    assert len(line) == 2
    dic[line[0]] = line[1]

pretrained_tokenizer = AutoTokenizer.from_pretrained("opus-mt-de-en")
pretrained_tokenizer_my = CharBPETokenizer(vocab='my-tokenizer/vocab.json', merges='my-tokenizer/merges.txt')

for x in dic:
    en = x
    my = dic[x]
    en_token = pretrained_tokenizer(en)['input_ids']
    if en_token[-1] == 0:
        en_token.pop()
    my_token = pretrained_tokenizer_my.encode(my).ids
    for token in my_token:
        dic_avg[token] += en_token

dic_emb = {}

import torch
import torch.nn as nn

bos_token = '<s>'
unk_token = '<unk>'
eos_token = '</s>'
pad_token = '<pad>'
special_tokens = [unk_token, bos_token, eos_token, pad_token]
special_tokens = set(special_tokens)

emb_name = 'my-fasttext-model.vec_underline'
emb_mt_name = 'opus-mt-de-en/embedding'
new_emb_name = f'{emb_name}_token_match_and_dic_avg'
vocab_size, emb_size = 50000, 512

dic_mt = {}
dic_mt_idx = []
for line in open(emb_mt_name):
    line = line.strip().split()
    if len(line) == 2:
        continue
    token = line[0]
    emb = line[1:]
    assert len(emb) == 512
    dic_mt[token] = emb
    dic_mt_idx.append(emb)
dic_mt_idx = np.array(dic_mt_idx, dtype='float')


import numpy as np

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
var = np.var(emb_list)
print(mean, var)
# 0.0013373970354963827 0.0022201915919596235
mean = np.mean(emb_list, axis=0)
# a = np.random.multivariate_normal(mean, cov, size=1).tolist()[0]
cov_arr = np.random.multivariate_normal(mean, cov, size=vocab_size)

cnt = 0
random_cnt = 0
with open(new_emb_name, 'w', encoding='utf-8') as f:
    f.write(str(vocab_size) + ' ' + str(emb_size) + '\n')
    for line in open(emb_name):
        line = line.strip().split()
        if len(line) == 2:
            continue
        token = line[0]
        # emb = line[1:]
        # assert len(emb) == 512
        if token in dic_mt:
            emb = dic_mt[token]
        else:
            my_token_id = cnt
            en_token_ids = dic_avg[my_token_id]
            if not en_token_ids:
                emb = [str(round(x, 7)) for x in cov_arr[my_token_id].tolist()]
                random_cnt += 1
            else:
                sum_emb = np.zeros_like(dic_mt_idx[0])
                for en_token_id in en_token_ids:
                    sum_emb += dic_mt_idx[en_token_id]
                avg_emb = sum_emb / len(en_token_ids)
                emb = [str(round(x, 7)) for x in avg_emb]
        assert len(emb) == 512
        f.write(' '.join([token] + emb))
        f.write('\n')
        cnt += 1
print(random_cnt)