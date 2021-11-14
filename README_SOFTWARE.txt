We have three child languages, My, Id, Tr.
We train a sub-word tokenizer using SentencePiece for each low-resource language, including My, Id and Tr. The tokenizers are trained on monolingual plain texts which are collected from Wikipedia's dumps(https://dumps.wikimedia.org). The toolkit wikiextractor(https://github.com/attardi/wikiextractor) is utilized to extract plain texts from the semi-structured data.

首先，先下载三种语言的维基百科语料，
wget --no-check-certificate https://dumps.wikimedia.org/mywiki/20210920/mywiki-20210920-pages-articles.xml.bz2
wget --no-check-certificate https://dumps.wikimedia.org/idwiki/20210920/idwiki-20210920-pages-articles.xml.bz2
wget --no-check-certificate https://dumps.wikimedia.org/trwiki/20210920/trwiki-20210920-pages-articles.xml.bz2

然后用wikiextractor抽取语料
python -m wikiextractor.WikiExtractor -b 1024M -o wiki_data {lang}wiki-20210920-pages-articles.xml.bz2

训练tokenizer
import sentencepiece as spm
spm.SentencePieceTrainer.train(f'--input={source_input_path} --model_prefix={lang}-tokenizer/spm --vocab_size=50000 --character_coverage=1.0 --hard_vocab_limit=false')

使用eflomal训练词对齐，训练数据采用train.{lang}和train.en，对齐方式采用grow-diag-final-and
python eflomal/align.py -i {lang}-en --model 3 -f {lang}-en.fwd -r {lang}-en.rev
atools -c grow-diag-final-and -i {lang}-en.fwd -j {lang}-en.rev > {lang}-en.sym

