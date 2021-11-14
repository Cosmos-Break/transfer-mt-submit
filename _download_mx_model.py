import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
ctx = mx.cpu()

wmt_model_name = 'transformer_en_de_512'
wmt_transformer_model, wmt_src_vocab, wmt_tgt_vocab = nlp.model.get_model(wmt_model_name,
                                                                        dataset_name='WMT2014',
                                                                        pretrained=True,
                                                                        ctx=ctx)

print(wmt_src_vocab.token_to_idx)
print(wmt_tgt_vocab.token_to_idx)