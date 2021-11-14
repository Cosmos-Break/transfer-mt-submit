lang=tr
mkdir ${lang}-tokenizer
# scp  mhxu@192.168.126.149:/home/mhxu/fastText-0.9.2/tr-fasttext-model.vec .
scp  mhxu@192.168.126.149:/home/mhxu/transfer-mt/vocab.json ${lang}-tokenizer/
scp  mhxu@192.168.126.149:/home/mhxu/transfer-mt/merges.txt ${lang}-tokenizer/

