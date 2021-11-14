import sys

bos_token = '<s>'
unk_token = '<unk>'
eos_token = '</s>'
pad_token = '<pad>'
special_tokens = [unk_token, bos_token, eos_token, pad_token]
special_tokens = set(special_tokens)

lang = sys.argv[1]
# lang = 'my'

emb_name = f'{lang}-fasttext-model.vec'
new_emb_name = f'{emb_name}_underline'
vocab_size, emb_size = 50000, 512
with open(new_emb_name, 'w', encoding='utf-8') as f, open(f'{lang}-tokenizer/vocab.txt', 'w') as fv:
    f.write(str(vocab_size) + ' ' + str(emb_size) + '\n')
    for line in open(emb_name):
        line = line.strip().split()
        if len(line) == 2:
            continue
        token = line[0]
        emb = line[1:]
        assert len(emb) == 512
        if token in special_tokens:
            fv.write(token + '\n')
            f.write(' '.join([token] + emb))
            f.write('\n')
            continue
        if token.endswith('</w>'):
            token = token[:-4]
        else:
            token = '▁' + token
        fv.write(token + '\n')
        f.write(' '.join([token] + emb))
        f.write('\n')