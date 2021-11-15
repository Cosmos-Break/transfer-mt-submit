# 根据sym文件输出词表
import nltk
import string
punc = string.punctuation
# nltk.download('punkt')
import sys
lang = sys.argv[1]
from nltk.tokenize import word_tokenize
# lang = 'my'
# lang = 'id'
# lang = 'tr'
L1, L2 = [], []
for line, line2 in zip(open(f'{lang}-en-data/train.en'), open(f'{lang}-en-data/train.{lang}')):
    if line == '\n' or line2 == '\n':
        continue
    L2.append(line.strip())
    L1.append(line2.strip())
# for line, line2 in zip(open(f'{lang}-en-data/val.en'), open(f'{lang}-en-data/val.{lang}')):
#     if line == '\n' or line2 == '\n':
#         continue
#     L1.append(line.strip())
#     L2.append(line2.strip())

# generate eflomal format data
# for l1, l2 in zip(L1, L2):
#     l1 = word_tokenize(l1)
#     l1 = ' '.join(l1)
#     l2 = word_tokenize(l2)
#     l2 = ' '.join(l2)
#     print(l1 + ' ||| ' + l2)

# build bilingual dic
import string
from collections import Counter, defaultdict
sym = []
dic = defaultdict(list)
for line in open(f'{lang}-en.sym'):
    sym.append(line.strip().split())

cnt = 0
for l1, l2, s in zip(L1, L2, sym):
    cnt += 1
    l1 = word_tokenize(l1)
    l2 = word_tokenize(l2)
    # l1 = l1.translate(str.maketrans('', '', string.punctuation)).split()
    # l2 = l2.translate(str.maketrans('', '', string.punctuation)).split()
    for x in s:
        a = x.split('-')
        a1 = int(a[0])
        a2 = int(a[1])
        if l1[a1] in punc or l2[a2] in punc:
            continue
        dic[l1[a1]].append(l2[a2])

with open(f'{lang}-en.dic', 'w') as f:
    for key, val in dic.items():
        cnt = Counter(val)
        common = cnt.most_common(1)[0]
        # if common[1] <= 1:
        #     continue
        f.write(key + ' ' + common[0] + '\n')
        # print(key, common[0])
