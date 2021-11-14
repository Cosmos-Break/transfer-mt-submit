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
print(pretrained_tokenizer)

model_checkpoint = "berteus-base-cased"
pretrained_tokenizer_eu = AutoTokenizer.from_pretrained(model_checkpoint)
pretrained_tokenizer_eu('123', add_special_tokens=False)

directory = 'eu-en-data'
def create_translation_data(
    source_input_path: str,
    target_input_path: str,
    output_path: str,
    delimiter: str = '\t',
    encoding: str = 'utf-8'
):
    """
    Creates the paired source and target dataset from the separated ones.
    e.g. creates `train.tsv` from `train.de` and `train.en`
    """
    with open(source_input_path, encoding=encoding) as f_source_in, \
         open(target_input_path, encoding=encoding) as f_target_in, \
         open(output_path, 'w', encoding=encoding) as f_out:

        for source_raw in f_source_in:
            source_raw = source_raw.strip()
            target_raw = f_target_in.readline().strip()
            if source_raw and target_raw:
                output_line = source_raw + delimiter + target_raw + '\n'
                f_out.write(output_line)

source_lang = 'eu'
target_lang = 'en'

data_files = {}
for split in ['train', 'val', 'test']:
    source_input_path = os.path.join(directory, f'{split}.{source_lang}')
    target_input_path = os.path.join(directory, f'{split}.{target_lang}')
    output_path = f'{split}.tsv'
    create_translation_data(source_input_path, target_input_path, output_path)
    data_files[split] = [output_path]

dataset_dict = load_dataset(
    'csv',
    delimiter='\t',
    column_names=[source_lang, target_lang],
    data_files=data_files
)

max_source_length = 128
max_target_length = 128
source_lang = "eu"
target_lang = "en"

cnt = 0
for source in dataset_dict['train']['eu']:
    model_input = pretrained_tokenizer(source, max_length=max_target_length, truncation=True)
    if 1 in model_input['input_ids']:
        tokens = pretrained_tokenizer.convert_ids_to_tokens(model_input['input_ids'])
        print(tokens)
        cnt += 1

print(cnt)