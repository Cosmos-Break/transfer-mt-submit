import json

# vocab_json = 'eu-tokenizer/vocab.json'
# vocab_json = 'my-tokenizer/vocab.json'
# vocab_json = 'id-tokenizer/vocab.json'
vocab_json = 'opus-mt-de-en/vocab.json'
vocab_txt = vocab_json[:-4] + 'txt'
data = json.load(open(vocab_json))
with open(vocab_txt, 'w') as f:
    for x in data:
        f.write(x + '\n')
