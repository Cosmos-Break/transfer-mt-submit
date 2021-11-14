# 根据sym文件输出词表
import nltk
import string
punc = string.punctuation
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
lang = 'my'
# lang = 'id'
# lang = 'tr'
# lang = 'eu'
# lang = 'az'
# lang = 'be'
# lang = 'sl'
L1, L2 = [], []
for line, line2 in zip(open(f'{lang}-en-data/train.en'), open(f'{lang}-en-data/train.{lang}')):
    if line == '\n' or line2 == '\n':
        continue
    L1.append(line.strip())
    L2.append(line2.strip())
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
with open(f'{lang}-en.dic', 'w') as f:
    for line in open(f'{lang}-en.priors'):
        line = line.strip().split()
        if line[0] == 'LEX':
            for _ in range(int(line[3])):
                dic[line[1]].append(line[2])
                # f.write(line[1] + ' ' + line[2] + '\n')
                # sym.append(line)
    for key in dic:
        val = Counter(dic[key]).most_common(1)[0][0]
        f.write(key + ' ' + val + '\n')
        
# cnt = 0
# for l1, l2, s in zip(L1, L2, sym):
#     cnt += 1
#     l1 = word_tokenize(l1)
#     l2 = word_tokenize(l2)
#     # l1 = l1.translate(str.maketrans('', '', string.punctuation)).split()
#     # l2 = l2.translate(str.maketrans('', '', string.punctuation)).split()
#     for x in s:
#         a = x.split('-')
#         a1 = int(a[0])
#         a2 = int(a[1])
#         if l1[a1] in punc or l2[a2] in punc:
#             continue
#         dic[l1[a1]].append(l2[a2])

# with open(f'en-{lang}.dic.all', 'w') as f:
#     for key, val in dic.items():
#         for x in val:
#             f.write(key + ' ' + x + '\n')
#             # print(key, x)

