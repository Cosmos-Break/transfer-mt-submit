./download_model.py en

./tools/extract-embed.sh transformer_en_de_512_WMT2014-e25287c5.params bpe.vocab.en source
./tools/extract-embed.sh cc.de.300.bin bpe.vocab.de target

wget http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/transformer_en_de_512_WMT2014-e25287c5.zip | unzip
https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/models/transformer_en_de_512_WMT2014-e25287c5.zip


wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.de.gz
wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.en.gz
gunzip corpus.tc.de.gz
gunzip corpus.tc.en.gz
curl http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/dev.tgz | tar xvzf -

pip install subword-nmt

subword-nmt learn-joint-bpe-and-vocab --input corpus.tc.de corpus.tc.en -s 30000 -o bpe.codes --write-vocabulary bpe.vocab.de bpe.vocab.en

subword-nmt apply-bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 < corpus.tc.de > corpus.tc.BPE.de
subword-nmt apply-bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 < corpus.tc.en > corpus.tc.BPE.en

subword-nmt apply-bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 < newstest2016.tc.de > newstest2016.tc.BPE.de
subword-nmt apply-bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 < newstest2016.tc.en > newstest2016.tc.BPE.en

# (Optional: run this if you have limited RAM on the training machine)
head -n 2000 corpus.tc.BPE.de.all > corpus.tc.BPE.de.tmp && mv corpus.tc.BPE.de.tmp corpus.tc.BPE.de
head -n 2000 corpus.tc.BPE.en.all > corpus.tc.BPE.en.tmp && mv corpus.tc.BPE.en.tmp corpus.tc.BPE.en

python -m sockeye.prepare_data \
                        -s corpus.tc.BPE.de \
                        -t corpus.tc.BPE.en \
                        -o train_data

python -m sockeye.train -d train_data \
                        -vs newstest2016.tc.BPE.de \
                        -vt newstest2016.tc.BPE.en \
                        --encoder rnn \
                        --decoder rnn \
                        --num-embed 256 \
                        --rnn-num-hidden 512 \
                        --rnn-attention-type dot \
                        --max-seq-len 60 \
                        --decode-and-evaluate 500 \
                        --use-cpu \
                        -o wmt_model


# English fastText Wikipedia embeddings
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.de.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.eu.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.sl.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.be.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.az.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.tr.vec


python MUSE/supervised.py --src_lang en --tgt_lang eu --src_emb wiki.en.vec --tgt_emb wiki.eu.vec --n_refinement 5 --dico_train identical_char --cuda False
python MUSE/unsupervised.py --src_lang en --tgt_lang eu --src_emb wiki.en.vec --tgt_emb wiki.eu.vec --n_refinement 5 --cuda False

wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.eu.vec
python MUSE/unsupervised.py --src_lang eu --tgt_lang en --src_emb wiki.eu.vec --tgt_emb transformer_en_de_512_WMT2014-e25287c5.params.src_embed.0.weight.vec --n_refinement 5 --cuda False

apt install git-lfs
git lfs install
git clone https://huggingface.co/ixa-ehu/berteus-base-cased
git clone https://huggingface.co/mrm8488/RoBasquERTa

!python MUSE/unsupervised.py --src_lang eu --tgt_lang en --src_emb berteus-base-cased/embedding --tgt_emb transformer_en_de_512_WMT2014-e25287c5.params.src_embed.0.weight.vec --n_refinement 5 --src_emb_dim 768 --emb_dim 512 --cuda True --dis_most_frequent 0

!python MUSE/supervised.py --src_lang eu --tgt_lang en --src_emb wiki.en.vec --tgt_emb wiki.eu.vec --n_refinement 5 --dico_train identical_char --cuda False


!python -m sockeye.replace_embedding -p transformer_en_de_512_WMT2014-e25287c5.params \
                                     -e vectors-eu.txt \
                                     -s source


transformer_en_de_512_WMT2014-e25287c5.params.source_embed_weight-vectors-eu.txt
vocab.src.0.vectors-eu.txt.json


python -m sockeye.train -s train.tags.eu-en.eu.eu \
                          -t train.tags.eu-en.en.en \
                          -vs IWSLT18.TED.dev2018.eu-en.eu.xml.eu \
                          -vt IWSLT18.TED.dev2018.eu-en.en.xml.en \
                          --source-noise-train \
                          --source-noise-permutation 3 \
                          --source-noise-deletion 0.1 \
                          --source-noise-insertion 0.1 \
                          --source-noise-insertion-vocab 50 \

0.
git clone https://huggingface.co/ixa-ehu/berteus-base-cased
git clone https://huggingface.co/Helsinki-NLP/opus-mt-de-en
python _download_huggingface_model.py
python _download_huggingface_mt_model.py
python dimensionality_reduction/codes/PCA/PCA.py
python _change_sharp_to_underline.py

# 目前第一版的vector-eu, 先PCA到512，再把#改成_，最后进行MUSE无监督对齐。


1.
!python MUSE/unsupervised.py --src_lang eu --tgt_lang de --src_emb berteus-base-cased/embedding_PCA_512_underline --tgt_emb opus-mt-de-en/embedding --n_refinement 5 --emb_dim 512 --cuda False --dis_most_frequent 0
python MUSE-my/unsupervised.py --src_lang eu --tgt_lang de --src_emb berteus-base-cased/embedding --tgt_emb opus-mt-de-en/embedding --n_refinement 5 --src_emb_dim 768 --emb_dim 512 --cuda False --dis_most_frequent 0 --epoch_size 1000 --n_epochs 1 
!python MUSE/supervised.py --src_lang eu --tgt_lang de --src_emb berteus-base-cased/embedding_PCA_512_underline --tgt_emb opus-mt-de-en/embedding --n_refinement 5 --emb_dim 512 --cuda True --dico_train /content/transfer-mt/_check_vocab_overlap



!python umwe/unsupervised.py --src_lang eu --tgt_lang de --src_emb berteus-base-cased/embedding_PCA_512_underline --tgt_emb opus-mt-de-en/embedding --n_refinement 5 --emb_dim 512 --device cuda --dis_most_frequent 0

!python _concat_emb.py > en-eu-concat-emb
!python _change_NMT_embedding.py


2.
git clone https://github.com/WebNLG/GenerationEval.git
cd GenerationEval
!apt install default-jre perl
./install_dependencies.sh
python3 eval.py -R data/en/references/reference -H data/en/hypothesis -nr 4 -m bleu,meteor,chrf++,ter,bert,bleurt
python3 eval.py -R data/en/references/reference -H data/en/hypothesis -nr 4 -m bleu,meteor,chrf++,ter,bert,bleurt


