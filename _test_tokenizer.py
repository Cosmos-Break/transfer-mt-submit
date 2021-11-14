import torch
import os

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerControl
from torch.utils.data import DataLoader
import time
import numpy as np
from tokenizers import CharBPETokenizer
from transformers import MarianTokenizer
from tokenizers import BPE

model_checkpoint = "opus-mt-de-en"
pretrained_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(pretrained_tokenizer)


pretrained_tokenizer_my = CharBPETokenizer(vocab='my-tokenizer/vocab.json', merges='my-tokenizer/merges.txt')


res = pretrained_tokenizer.tokenize('''The murder probably took place on Sunday, July 31 according to the local police, which also states that at least ten suspects have been questioned.''')

model_inputs = pretrained_tokenizer_my.encode("ဒေသခံ ရဲတွေ အဆိုအရ ဇူလိုင် ၃၁ ရက်နေ့ တနင်္ဂနွေနေ့ မှာ လူသတ်မှု ဖြစ်ပွားခဲ့တာ ဖြစ်နိုင် ပြီး အနည်းဆုံး သံသယရှိသူ ၁၀ ယောက် ကို မေးမြန်းပြီးပြီ လို့ လည်းပဲ ဖော်ပြပါတယ် ။")

print(res)

print(model_inputs.tokens)
['▁The', '▁', 'mur', 'der', '▁pro', 'b', 'ab', 'ly', '▁to', 'ok', '▁', 'place', '▁on', '▁Sun', 'day', ',', '▁Ju', 'ly', '▁31', '▁', 'ac', 'cord', 'ing', '▁to', '▁the', '▁', 'local', '▁', 'pol', 'ice', ',', '▁wh', 'ich', '▁also', '▁', 'state', 's', '▁that', '▁at', '▁le', 'ast', '▁ten', '▁su', 'spect', 's', '▁have', '▁be', 'en', '▁', 'quest', 'ion', 'ed', '.']
['ဒေသခံ</w>', 'ရဲ', 'တွေ</w>', 'အဆိုအရ</w>', 'ဇူလိုင်</w>', '၃၁</w>', 'ရက်နေ့</w>', 'တနင်္ဂနွေနေ့</w>', 'မှာ</w>', 'လူသတ်မှု</w>', 'ဖြစ်ပွား', 'ခဲ့တာ</w>', 'ဖြစ်နိုင်</w>', 'ပြီး</w>', 'အနည်းဆုံး</w>', 'သံသယ', 'ရှိသူ</w>', '၁၀</w>', 'ယောက်</w>', 'ကို</w>', 'မေးမြန်း', 'ပြီးပြီ</w>', 'လို့</w>', 'လည်း', 'ပဲ</w>', 'ဖော်ပြ', 'ပါတယ်</w>', '။</w>']