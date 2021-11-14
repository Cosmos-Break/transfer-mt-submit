L1, L2 = [], []
L1_word_to_id = {}
lang = 'eu'

for index, line in enumerate(open('opus-mt-de-en/vocab.txt')):
    line = line.strip()
    L1_word_to_id[line] = index
    L1.append(line)
for line in open(f'{lang}-tokenizer/vocab.txt'):
    line = line.strip()
    L2.append(line)

# 输出相同vocab的id
for index2, token2 in enumerate(L2):
    if token2 in L1_word_to_id:
        pass
        # print(L1_word_to_id[token2], index2)

# for index1, token1 in enumerate(L1):
#     for index2, token2 in enumerate(L2):
#         if token1 == token2:
#             print(index1, index2)
            
# 输出没有重叠的token id, eu
# L2s = set(L2)
# for index, token in enumerate(L1):
#     if token not in L2s:
#         print(index)

# 输出没有重叠的token id, de
# L1s = set(L1)
# for index, token in enumerate(L2):
#     if token not in L1s:
#         print(index)


# L1 = set(L1)
# L2 = set(L2)
# cnt = 0
# tokens = set()
# for token in L2:
#     if token in L1:
#         token = token.lower()
#         tokens.add(token)

# 输出所有
# for token in tokens:
#     print(token+' '+token)
#     cnt += 1

# 输出数字
# for token in tokens:
#     if token.isdigit():
#         print(token+' '+token)
#         cnt += 1

# de-de 重叠和不重叠
overlap_set = set()
with open('overlap_tokenid_de_de', 'w') as f1, open('not_overlap_tokenid_de_de', 'w') as f2:
    for line in open('de-de.dic_orig'):
        line = line.strip().split()[0]
        if line in L1_word_to_id:
            f1.write(' '.join([str(L1_word_to_id[line]), str(L1_word_to_id[line]),'\n']))
            overlap_set.add(L1_word_to_id[line])
    for line in L1_word_to_id:
        if L1_word_to_id[line] not in overlap_set:
            f2.write(str(L1_word_to_id[line]) + '\n')