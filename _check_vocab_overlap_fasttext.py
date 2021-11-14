bos_token = '<s>'
unk_token = '<unk>'
eos_token = '</s>'
pad_token = '<pad>'
special_tokens = [unk_token, bos_token, eos_token, pad_token]
special_tokens = set(special_tokens)


L1, L2 = [], []
for line in open('my-tokenizer/vocab.txt'):
    line = line.strip()
    if line in special_tokens:
        continue
    L1.append(line)

for line in open('opus-mt-de-en/vocab.txt'):
    line = line.strip()
    if line in special_tokens:
        continue
    L2.append(line)

L1 = set(L1)
L2 = set(L2)
cnt = 0
tokens = set()
for token in L2:
    if token in L1:
        token = token.lower()
        tokens.add(token)

# 输出所有
for token in tokens:
    print(token+' '+token)
    cnt += 1

# 输出数字
# for token in tokens:
#     if token.isdigit():
#         print(token+' '+token)
#         cnt += 1

# print(cnt)