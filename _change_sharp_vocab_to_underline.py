special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
special_tokens = set(special_tokens)

vocab_name = 'de_vocab.txt'
new_vocab_name = f'{vocab_name}_underline'
with open(new_vocab_name, 'w', encoding='utf-8') as f:
    for line in open(vocab_name):
        token = line.strip()
        if token in special_tokens:
            f.write(token)
            f.write('\n')
            continue
        if token.startswith('##'):
            token = token[2:]
        else:
            token = '▁' + token
        f.write(token)
        f.write('\n')