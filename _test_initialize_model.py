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

model_checkpoint = "opus-mt-de-en"
pretrained_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_model = torch.load('opus-mt-de-en/pytorch_model_concat_emb.bin').to(device)
print(pretrained_tokenizer)
pretrained_model.init_weights()


