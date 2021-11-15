import sys
lang = sys.argv[1]
# lang = 'my'
with open(f'{lang}-fasttext-model.vec_underline', 'w') as f, open(f'{lang}-tokenizer/vocab.txt', 'w') as fv:
    f.write('50000 512\n')
    for line in open(f'{lang}-tokenizer/spm.vocab'):
        token = line.strip().split()[0]
        fv.write(token + '\n')
        f.write(' '.join([token] + ['0.0'] * 512) + '\n')
    