time=$(date "+%Y%m%d-%H%M%S")
lang='tr'

for i in $(seq 1 5)
do
    topk=$i

    # topk-mean
    # python -u _token_matching-dic-avg-emb_topk_mean.py ${lang} ${topk}
    # python -u _concat_emb.py ${lang}-fasttext-model.vec_underline_token_match_and_dic_avg > en-${lang}-concat-emb
    # python -u _change_NMT_embedding.py en-${lang}-concat-emb

    # topk
    # python -u _token_matching-dic-avg-emb_topk.py ${lang} ${topk}
    # python -u _concat_emb.py ${lang}-fasttext-model.vec_underline_token_match_and_dic_avg > en-${lang}-concat-emb
    # python -u _change_NMT_embedding.py en-${lang}-concat-emb

    # dic_avg
    python _token_matching-dic-avg-emb.py ${lang}
    python _concat_emb.py ${lang}-fasttext-model.vec_underline_token_match_and_dic_avg > en-${lang}-concat-emb
    python _change_NMT_embedding.py en-${lang}-concat-emb

    # dic_all_avg
    # python _token_matching-dic-avg-emb_all.py ${lang}
    # python _concat_emb.py ${lang}-fasttext-model.vec_underline_token_match_and_dic_avg > en-${lang}-concat-emb
    # python _change_NMT_embedding.py en-${lang}-concat-emb


    bsub -n 1 -q HPC.S1.GPU.X785.sha -o logs/train_${lang}_top${topk}_${time}.log -gpu num=1:mode=exclusive_process python -u train.py ${lang}
    echo train_${lang}_top${topk}_${time}.log
    sleep 300
done
