en_de_tokens = set()
for line in open('opus-mt-de-en/vocab.txt'):
    line = line.strip()
    en_de_tokens.add(line)

lang = 'eu'
with open(f'en-{lang}.dic.same_token', 'w') as f:
    for line in open(f'{lang}-tokenizer/vocab.txt'):
        line = line.strip()
        if line in en_de_tokens:
            f.write(' '.join([line.lower(), line.lower(), '\n']))
            