from importlib import import_module
import torch
from torch import nn
import numpy as np
import sys
from _partial_freeze import MyMarianEncoder
concat_emb_name = sys.argv[1]
# concat_emb_name = 'my-fasttext-model.vec_underline_token_match_and_dic_avg'
trainable_weights = []
for line in open(concat_emb_name):
    line = line.strip().split()
    if len(line) == 2:
        continue
    line = line[1:]
    line = [float(x) for x in line]
    assert len(line) == 512
    trainable_weights.append(np.array(line))

dic_size, emb_size = len(trainable_weights), len(trainable_weights[0])

trainable_weights = np.array(trainable_weights)
trainable_weights = torch.FloatTensor(trainable_weights)
# trainable_weights_T = trainable_weights.transpose(-1, 0)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_path = 'opus-mt-de-en'
vocab_path = f'{model_path}/vocab.txt'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

trainable_emb_layer = nn.Embedding.from_pretrained(trainable_weights, freeze=False, padding_idx=3)

# model.model.shared = pretrained_emb_layer
# embedding = model.model.shared.weight.data.copy_(torch.FloatTensor(trainable_weights))
# model.model.encoder.embed_tokens = pretrained_emb_layer
# model.model.encoder.embed_tokens = trainable_emb_layer
print('# of parameters: ', model.num_parameters())
myencoder = MyMarianEncoder(model.config)

pretrained_emb_layer = nn.Embedding.from_pretrained(model.model.encoder.embed_tokens.weight, freeze=True, padding_idx=58100)
myencoder.embed_tokens = pretrained_emb_layer
myencoder.layers = model.model.encoder.layers
myencoder.trainable_embedding = trainable_emb_layer
model.model.encoder = myencoder
print('# of parameters: ', model.num_parameters())
# decoder embedding也不用改?
# model.model.decoder.embed_tokens = pretrained_emb_layer

# lm_head不用改?
# model.lm_head = nn.Linear(emb_size, dic_size, bias=False)
# model.lm_head.weight.data.copy_(trainable_weights)
torch.save(model, 'opus-mt-de-en/pytorch_model_concat_emb.bin')
