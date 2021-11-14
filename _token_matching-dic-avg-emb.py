from collections import defaultdict
import numpy as np
from numpy.random import random
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTokenizedInput
from tokenizers import CharBPETokenizer
from tokenizers.pre_tokenizers import Whitespace
from transformers.utils.dummy_pt_objects import SqueezeBertModel
import sentencepiece as spm
pretrained_tokenizer_my = spm.SentencePieceProcessor()

np.random.seed(42)
import sys
lang = sys.argv[1]
dic = defaultdict(list)
dic_avg = defaultdict(list)
for line in open(f'{lang}-en.dic'):
    line = line.strip().split()
    assert len(line) == 2
    # dic[line[1]].append(line[0])
    dic[line[0]].append(line[1])
    # dic[line[0]] = line[1]

pretrained_tokenizer = AutoTokenizer.from_pretrained("opus-mt-de-en")
# pretrained_tokenizer_my = CharBPETokenizer(vocab=f'{lang}-tokenizer/vocab.json', merges=f'{lang}-tokenizer/merges.txt')
pretrained_tokenizer_my.load(f'{lang}-tokenizer/spm.model')

for x in dic:
    my = x
    en = dic[x]
    try:
        en_tokens = pretrained_tokenizer(en, add_special_tokens=False)['input_ids']
    except:
        continue
    my_token = pretrained_tokenizer_my.encode_as_ids(my)
    for token in my_token:
        for en_token in en_tokens:
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

emb_name = f'{lang}-fasttext-model.vec_underline'
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
                mean = 0
                std = 0.05
                # mean = 0.0013373970354963827
                # std = 0.04711890907013472
                emb = [str(round(x, 7)) for x in np.random.normal(mean, std, 512).tolist()]
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