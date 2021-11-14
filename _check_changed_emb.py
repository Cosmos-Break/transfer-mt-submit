embedding = pretrained_model.model.encoder.embed_tokens.weight
n, emb_size = embedding.shape
# for x in embedding[:10].tolist():
#     print(x)

embedding_file = "trained_embedding"
with open(embedding_file, 'w', encoding='utf-8') as f:
    f.write(str(n) + ' ' + str(emb_size) + '\n')
    for i in range(n):
        emb = [str(round(x, 7)) for x in embedding[i].tolist()]
        f.write(' '.join(emb))
        f.write('\n')
!tail -n 50000 trained_embedding > trained_embedding_my
orig_emb = []
trained_emb = []
for line in open('trained_embedding_my'):
    trained_emb.append(line.strip())
for line in open('my-fasttext-model.vec_underline_token_match_and_dic_avg'):
    line = line.strip().split()
    if len(line) == 2:
        continue
    emb = line[1:]
    assert len(emb) == 512
    emb = ' '.join(emb)
    orig_emb.append(emb)
cnt = 0
for x, y in zip(orig_emb, trained_emb):
    if x == y:
        cnt += 1
        print(x)
print(cnt)