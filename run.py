lang = 'my'
wiki_directory = 'train'
import sentencepiece as spm
import os
import subprocess
# subprocess.run(f'', shell=True)

# 下载wiki数据
# subprocess.run(f'wget --no-check-certificate https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2', shell=True)
# subprocess.run(f'python -m wikiextractor.WikiExtractor -b 1024M -o wiki_data {lang}wiki-latest-pages-articles.xml.bz2', shell=True)
# subprocess.run(f'mkdir {wiki_directory}', shell=True)
# subprocess.run(f'cp wiki_data/AA/wiki_00 train/train.{lang}', shell=True)


# 训练tokenzier
source_input_path = os.path.join(wiki_directory, f'train.{lang}')
print(source_input_path)
subprocess.run(f'mkdir {lang}-tokenizer', shell=True)
spm.SentencePieceTrainer.train(f'--input={source_input_path} --model_prefix={lang}-tokenizer/spm --vocab_size=50000 --character_coverage=1.0 --hard_vocab_limit=false')

# eflomal对齐
import nltk
import string
punc = string.punctuation
nltk.download('punkt')
from nltk.tokenize import word_tokenize
lang = 'id'
L1, L2 = [], []
for line, line2 in zip(open(f'{lang}-en-data/train.en'), open(f'{lang}-en-data/train.{lang}')):
    if line == '\n' or line2 == '\n':
        continue
    L1.append(line.strip())
    L2.append(line2.strip())
    
# generate eflomal format data
with open(f'eflomal_data.{lang}-en', 'w') as f:
    for l1, l2 in zip(L1, L2):
        l1 = word_tokenize(l1)
        l1 = ' '.join(l1)
        l2 = word_tokenize(l2)
        l2 = ' '.join(l2)
        f.write(l2 + ' ||| ' + l1 + '\n')