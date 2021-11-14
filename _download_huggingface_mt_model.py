from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel
import json
import torch
model_path = 'opus-mt-de-en'
vocab_path = f'{model_path}/vocab.txt'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# vocab = json.load(open('opus-mt-de-en/vocab.json'))
vocab = tokenizer.decoder
with open(vocab_path, 'w') as f:
    for i in range(len(vocab)):
        f.write(vocab[i] + '\n')


embedding = model.model.shared.weight
n, emb_size = embedding.shape
# for x in embedding[:10].tolist():
#     print(x)

embedding_file = f"{model_path}/embedding"
with open(embedding_file, 'w', encoding='utf-8') as f:
    f.write(str(n) + ' ' + str(emb_size) + '\n')
    for i in range(n):
        emb = [str(round(x, 7)) for x in embedding[i].tolist()]
        f.write(' '.join([vocab[i]] + emb))
        f.write('\n')