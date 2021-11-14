# 根据sym文件输出词表
import nltk
import string
punc = string.punctuation
# nltk.download('punkt')
import sys
# lang = sys.argv[1]
from nltk.tokenize import word_tokenize
# lang = 'my'
# lang = 'id'
lang = 'tr'

from transformers import AutoTokenizer
from tokenizers import CharBPETokenizer
from tokenizers.pre_tokenizers import Whitespace
import sentencepiece as spm
model_checkpoint = "opus-mt-de-en"
pretrained_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
pretrained_tokenizer_L2 = spm.SentencePieceProcessor()
pretrained_tokenizer_L2.load(f'{lang}-tokenizer/spm.model')

L1, L2 = [], []
for line, line2 in zip(open(f'{lang}-en-data/train.en'), open(f'{lang}-en-data/train.{lang}')):
    if line == '\n' or line2 == '\n':
        continue
    line = line.strip()
    line2 = line2.strip()
    line = ' '.join(pretrained_tokenizer.tokenize(line))
    line2 = ' '.join(pretrained_tokenizer_L2.encode_as_pieces(line2))
    L1.append(line)
    L2.append(line2)

# build bilingual dic
import string
from collections import Counter, defaultdict
sym = []
dic = defaultdict(list)
for line in open(f'{lang}-en-subword.sym'):
    sym.append(line.strip().split())

cnt = 0
# L2是小语种，L1是英文
for l1, l2, s in zip(L2, L1, sym):
    cnt += 1
    l1 = l1.split()
    l2 = l2.split()
    for x in s:
        a = x.split('-')
        a1 = int(a[0])
        a2 = int(a[1])
        if l1[a1] in punc or l2[a2] in punc:
            continue
        dic[l1[a1]].append(l2[a2])

with open(f'{lang}-en-subword.dic', 'w') as f:
    for key, val in dic.items():
        cnt = Counter(val)
        common = cnt.most_common(1)[0]
        # if common[1] <= 1:
        #     continue
        f.write(key + ' ' + common[0] + '\n')
        # print(key, common[0])
