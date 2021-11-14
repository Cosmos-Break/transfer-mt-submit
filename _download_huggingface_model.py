from transformers import AutoTokenizer, AutoModel
model_path = 'berteus-base-cased'
vocab_path = f'{model_path}/vocab.txt'

vocab = []
with open(vocab_path) as f:
    for line in f:
        vocab.append(line.strip())

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
embedding = model.embeddings.word_embeddings.weight
n, emb_size = embedding.shape
embedding_file = f"{model_path}/embedding"
with open(embedding_file, 'w', encoding='utf-8') as f:
    f.write(str(n) + ' ' + str(emb_size) + '\n')
    for i in range(n):
        emb = [str(round(x, 7)) for x in embedding[i].tolist()]
        f.write(' '.join([vocab[i]] + emb))
        f.write('\n')
print('done')