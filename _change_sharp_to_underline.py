emb_name = 'berteus-base-cased/embedding_PCA_512'
new_emb_name = f'{emb_name}_underline'
with open(new_emb_name, 'w', encoding='utf-8') as f:
    for line in open(emb_name):
        line = line.strip().split()
        if len(line) == 2:
            f.write(' '.join(line))
            f.write('\n')
            continue
        token = line[0]
        emb = line[1:]
        assert len(emb) == 512
        if token.startswith('##'):
            token = '▁' + token[2:]
        f.write(' '.join([token] + emb))
        f.write('\n')