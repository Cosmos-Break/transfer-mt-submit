import random
f1 = 'id-en-data/en.all'
f2 = 'id-en-data/id.all'
en_id = []
for line1, line2 in zip(open(f1), open(f2)):
    en_id.append((line1, line2))
random.shuffle(en_id)
with open('id-en-data/test.en', 'w') as fen, open('id-en-data/test.id', 'w') as fid:
    for x in en_id[:1000]:
        fen.write(x[0])
        fid.write(x[1])

with open('id-en-data/val.en', 'w') as fen, open('id-en-data/val.id', 'w') as fid:
    for x in en_id[1000:2000]:
        fen.write(x[0])
        fid.write(x[1])

with open('id-en-data/train.en', 'w') as fen, open('id-en-data/train.id', 'w') as fid:
    for x in en_id[2000:]:
        fen.write(x[0])
        fid.write(x[1])