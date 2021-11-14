import sys
import json
lang = sys.argv[1]
# lang = 'my'
with open(f'{lang}-fasttext-model.vec', 'w') as f:
    vocab = json.load(open(f'{lang}-tokenizer/vocab.json'))
    f.write('50000 512\n')
    for v in vocab:
        f.write(' '.join([v] + ['0.0'] * 512) + '\n')
    