import random
import sys
dic = {}
for line in open('en-my.dic'):
    line = line.strip().split()
    w1, w2 = line
    w1 = list(w1)
    w2 = list(w2)
    w1[0] = w1[0].upper()
    w2[0] = w2[0].upper()
    dic[''.join(w1)] = ''.join(w2)
    w1[0] = w1[0].lower()
    w2[0] = w2[0].lower()
    dic[''.join(w1)] = ''.join(w2)


fen = open('_generate_corpus.en', 'w')
feu = open('_generate_corpus.my', 'w')
cnt = 0
prob = float(sys.argv[1]) # 替换概率
# prob = 0.2 # 替换概率
for line in  open('my-en-data/train.en'):
    line_orig = line
    fen.write(line_orig)
    line = line.strip().split()
    n = len(line)
    for i in range(n):
        if line[i] in dic:
            # random substitute
            if random.random() < prob:
                line[i] = dic[line[i]]
                cnt += 1
    line = ' '.join(line)
    feu.write(line + '\n')

print(cnt)

fen.close()
feu.close()