import torch
from torch import nn
import numpy as np
import sys
concat_emb_name = sys.argv[1]
pretrained_weights = []
for line in open(concat_emb_name):
    line = line.strip().split()
    line = [float(x) for x in line]
    assert len(line) == 512
    pretrained_weights.append(np.array(line))

dic_size, emb_size = len(pretrained_weights), len(pretrained_weights[0])

pretrained_weights = np.array(pretrained_weights)
pretrained_weights = torch.FloatTensor(pretrained_weights)
# pretrained_weights_T = pretrained_weights.transpose(-1, 0)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_path = 'opus-mt-de-en'
vocab_path = f'{model_path}/vocab.txt'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

pretrained_emb_layer = nn.Embedding.from_pretrained(pretrained_weights, freeze=False, padding_idx=58100)
# model.model.shared = pretrained_emb_layer
# embedding = model.model.shared.weight.data.copy_(torch.FloatTensor(pretrained_weights))
model.model.encoder.embed_tokens = pretrained_emb_layer

# decoder embedding也不用改?
# model.model.decoder.embed_tokens = pretrained_emb_layer

# lm_head不用改?
# model.lm_head = nn.Linear(emb_size, dic_size, bias=False)
# model.lm_head.weight.data.copy_(pretrained_weights)
torch.save(model, 'opus-mt-de-en/pytorch_model_concat_emb.bin')
