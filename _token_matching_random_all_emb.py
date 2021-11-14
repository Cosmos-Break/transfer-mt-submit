import numpy as np
np.random.seed(42)
import sys
lang = sys.argv[1]

emb_name = f'{lang}-fasttext-model.vec_underline'
emb_mt_name = 'opus-mt-de-en/embedding'
new_emb_name = f'{emb_name}_token_match_random_all_emb'
vocab_size, emb_size = 50000, 512

with open(new_emb_name, 'w', encoding='utf-8') as f:
    f.write(str(vocab_size) + ' ' + str(emb_size) + '\n')
    for line in open(emb_name):
        line = line.strip().split()
        if len(line) == 2:
            continue
        token = line[0]
        mean = 0
        std = 0.05
        # mean = 0.0013373970354963827
        # std = 0.04711890907013472
        emb = [str(round(x, 7)) for x in np.random.normal(mean, std, 512).tolist()]
        assert len(emb) == 512
        f.write(' '.join([token] + emb))
        f.write('\n')