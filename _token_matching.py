import numpy as np
import sys
lang = sys.argv[1]

bos_token = '<s>'
unk_token = '<unk>'
eos_token = '</s>'
pad_token = '<pad>'
special_tokens = [unk_token, bos_token, eos_token, pad_token]
special_tokens = set(special_tokens)

emb_name = f'{lang}-fasttext-model.vec_underline'
emb_mt_name = 'opus-mt-de-en/embedding'
new_emb_name = f'{emb_name}_token_match'
vocab_size, emb_size = 50000, 512

dic_mt = {}
for line in open(emb_mt_name):
    line = line.strip().split()
    if len(line) == 2:
        continue
    token = line[0]
    emb = line[1:]
    assert len(emb) == 512
    dic_mt[token] = emb

cnt = 0
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
            emb = [str(round(x, 7)) for x in np.random.normal(0, 0.05, 512).tolist()]
            cnt += 1
        assert len(emb) == 512
        f.write(' '.join([token] + emb))
        f.write('\n')
print(cnt)