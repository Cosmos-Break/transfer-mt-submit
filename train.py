import os
import math
import time
import torch
import random
import sys
import numpy as np
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
from tokenizers import ByteLevelBPETokenizer
import subprocess

lang = sys.argv[1]
directory = f'{lang}-en-data'
def create_translation_data(
    source_input_path: str,
    target_input_path: str,
    output_path: str,
    delimiter: str = '\x01',
    encoding: str = 'utf-8'
):
    """
    Creates the paired source and target dataset from the separated ones.
    e.g. creates `train.tsv` from `train.de` and `train.en`
    """
    with open(source_input_path, encoding=encoding) as f_source_in, \
         open(target_input_path, encoding=encoding) as f_target_in, \
         open(output_path, 'w', encoding=encoding) as f_out:
        for source_raw, target_raw in zip(f_source_in, f_target_in):
            source_raw = source_raw.strip()
            target_raw = target_raw.strip()
            if source_raw and target_raw:
                output_line = source_raw + delimiter + target_raw + '\n'
                f_out.write(output_line)

source_lang = lang
target_lang = 'en'

data_files = {}
for split in ['train', 'val', 'test']:
    source_input_path = os.path.join(directory, f'{split}.{source_lang}')
    target_input_path = os.path.join(directory, f'{split}.{target_lang}')
    output_path = f'{split}.tsv'
    create_translation_data(source_input_path, target_input_path, output_path)
    data_files[split] = [output_path]


subprocess.run('shuf test.tsv -o test.tsv', shell=True)
subprocess.run('shuf val.tsv -o val.tsv', shell=True)
subprocess.run('shuf train.tsv -o train.tsv', shell=True)

dataset_dict = load_dataset(
    'csv',
    delimiter=r'\x01',
    column_names=[source_lang, target_lang],
    data_files=data_files
)

from transformers import AutoTokenizer
from tokenizers import CharBPETokenizer
from tokenizers.pre_tokenizers import Whitespace
import sentencepiece as spm



model_checkpoint = "opus-mt-de-en"
pretrained_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(pretrained_tokenizer)

pretrained_tokenizer_L2 = spm.SentencePieceProcessor()
pretrained_tokenizer_L2.load(f'{lang}-tokenizer/spm.model')
print(pretrained_tokenizer_L2)
# pretrained_tokenizer_L2 = CharBPETokenizer(f'{lang}-tokenizer/vocab.json', f'{lang}-tokenizer/merges.txt')

max_source_length = 128
max_target_length = 128
# orig_emb_size
orig_emb_size = 58101
pad_token_id = 58100
def batch_tokenize_fn(examples):
    """
    Generate the input_ids and labels field for huggingface dataset/dataset dict.
    
    Truncation is enabled, so we cap the sentence to the max length, padding will be done later
    in a data collator, so pad examples to the longest length in the batch and not the whole dataset.
    """
    sources = examples[source_lang]
    targets = examples[target_lang]
    model_inputs = {}
    # for encoding in pretrained_tokenizer_L2.encode_as_ids(sources):
    #     print(encoding)
    #     input_ids = encoding
    #     attention_mask = [1] * len(input_ids)
    #     model_inputs['input_ids'] = input_ids
    #     model_inputs['attention_mask'] = attention_mask
    input_ids = pretrained_tokenizer_L2.encode_as_ids(sources)
    attention_mask = [1] * len(input_ids)
    model_inputs['input_ids'] = input_ids
    model_inputs['attention_mask'] = attention_mask
    n = len(model_inputs['input_ids'])
    for i in range(n):
        model_inputs['input_ids'][i] += orig_emb_size
    if n != max_target_length:
        model_inputs['input_ids'].append(0)
        model_inputs['attention_mask'].append(1)
    else:
        model_inputs['input_ids'][n-1] = 0

    # setup the tokenizer for targets,
    # huggingface expects the target tokenized ids to be stored in the labels field
    with pretrained_tokenizer.as_target_tokenizer():
        labels = pretrained_tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset_dict = dataset_dict.map(batch_tokenize_fn)

class Seq2SeqDataCollator:
    def __init__(
        self,
        max_length: int,
        pad_token_id: int,
        pad_label_token_id: int = -100
    ):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.pad_label_token_id = pad_label_token_id
        
    def __call__(self, batch):
        source_batch = []
        source_len = []
        target_batch = []
        target_len = []
        for example in batch:
            source = example['input_ids']
            source_len.append(len(source))
            source_batch.append(source)

            target = example['labels']
            target_len.append(len(target))
            target_batch.append(target)

        source_padded = self.process_encoded_text(source_batch, source_len, self.pad_token_id)
        target_padded = self.process_encoded_text(target_batch, target_len, self.pad_label_token_id)
        attention_mask = generate_attention_mask(source_padded, self.pad_token_id)
        return {
            'input_ids': source_padded,
            'labels': target_padded,
            'attention_mask': attention_mask
        }

    def process_encoded_text(self, sequences, sequences_len, pad_token_id):
        sequences_max_len = np.max(sequences_len)
        max_length = min(sequences_max_len, self.max_length)
        padded_sequences = pad_sequences(sequences, max_length, pad_token_id)
        return torch.LongTensor(padded_sequences)


def generate_attention_mask(input_ids, pad_token_id):
    return (input_ids != pad_token_id).long()

    
def pad_sequences(sequences, max_length, pad_token_id):
    """
    Pad the list of sequences (numerical token ids) to the same length.
    Sequence that are shorter than the specified ``max_len`` will be appended
    with the specified ``pad_token_id``. Those that are longer will be truncated.

    Parameters
    ----------
    sequences : list[int]
        List of numerical token ids.

    max_length : int
         Maximum length that all sequences will be truncated/padded to.

    pad_token_id : int
        Padding token index.

    Returns
    -------
    padded_sequences : 1d ndarray
    """
    num_samples = len(sequences)
    padded_sequences = np.full((num_samples, max_length), pad_token_id)
    for i, sequence in enumerate(sequences):
        sequence = np.array(sequence)[:max_length]
        padded_sequences[i, :len(sequence)] = sequence

    return padded_sequences

from transformers import AutoModelForSeq2SeqLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_model = torch.load('opus-mt-de-en/pytorch_model_concat_emb.bin').to(device)
# pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)
print('# of parameters: ', pretrained_model.num_parameters())

# modules = [pretrained_model.model.decoder, pretrained_model.lm_head, pretrained_model.model.shared, pretrained_model.model.encoder.embed_tokens]
modules = [pretrained_model.model.decoder, pretrained_model.lm_head, pretrained_model.model.shared]
# modules = [pretrained_model.model.decoder.layers]
for module in modules:
    for name, param in module.named_parameters():
        param.requires_grad = False

def generate_translation(model, tokenizer, example):
    """print out the source, target and predicted raw text."""
    source = example[source_lang]
    target = example[target_lang]
    input_ids = example['input_ids']
    input_ids = torch.LongTensor(input_ids).view(1, -1).to(model.device)
    print('input_ids: ', input_ids)
    generated_ids = model.generate(input_ids)
    print('generated_ids: ', generated_ids)
    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print('source: ', source)
    print('target: ', target)
    print('prediction: ', prediction)


now = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
# checkpoint_path = f'/content/drive/Shareddrives/Aria3/checkpoint-{now}'
checkpoint_path = f'./{lang}-checkpoint-{now}'
# checkpoint_path = './checkpoint'
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
batch_size = 32
args = Seq2SeqTrainingArguments(
    output_dir=checkpoint_path,
    evaluation_strategy="steps",
    learning_rate=0.00005,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=15,
    num_train_epochs=200,
    load_best_model_at_end=True,
    predict_with_generate=True,
    remove_unused_columns=True,
    fp16=True,
    gradient_accumulation_steps=2,
    eval_steps=500,
    warmup_steps=100,
    dataloader_pin_memory=False
)

data_collator = Seq2SeqDataCollator(max_source_length, pad_token_id)
callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
trainer = Seq2SeqTrainer(
    pretrained_model,
    args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset_dict["train"],
    eval_dataset=tokenized_dataset_dict["val"],
    callbacks=callbacks
)

trainer = Seq2SeqTrainer(
    pretrained_model,
    args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset_dict["train"],
    eval_dataset=tokenized_dataset_dict["val"],
    callbacks=callbacks
)

# trainer_output = trainer.train('/content/drive/Shareddrives/Aria3/my-checkpoint-token-match-random-avg_emb_all_dataAUG/checkpoint-5500')
trainer_output = trainer.train()

for checkpoint in os.listdir(checkpoint_path):
    if 'checkpoint-' in checkpoint:
        print(checkpoint)
        # if int(checkpoint.split('-')[1]) < 6500:
        #     continue
        PATH = f'{checkpoint_path}/{checkpoint}/pytorch_model.bin'
        print(PATH)
        pretrained_model.load_state_dict(torch.load(PATH), strict=False)
        data_collator = Seq2SeqDataCollator(max_source_length, pad_token_id, pad_token_id)
        data_loader = DataLoader(tokenized_dataset_dict['test'], collate_fn=data_collator, batch_size=32, shuffle=True)
        target_list = []
        prediction_list = []
        for example in data_loader:
            input_ids = example['input_ids']
            # input_ids = torch.LongTensor(input_ids).to(model.device)
            # input_ids = torch.LongTensor([item.cpu().detach().numpy() for item in input_ids]).to(pretrained_model.device)
            # print('input_ids: ', input_ids)
            generated_ids = pretrained_model.generate(input_ids.to(pretrained_model.device))
            generated_ids = generated_ids.detach().cpu().numpy()
            predictions = pretrained_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            labels = example['labels'].detach().cpu().numpy()
            targets = pretrained_tokenizer.batch_decode(labels, skip_special_tokens=True)
            target_list += targets
            prediction_list += predictions
        
        with open(f'{PATH}-target', 'w') as f:
            for line in target_list:
                f.write(line + '\n')
        with open(f'{PATH}-prediction', 'w') as f:
            for line in prediction_list:
                f.write(line + '\n')
        
        subprocess.run(f'python eval_Generation.py -R {PATH}-target -H {PATH}-prediction -nr 1 -m bleu', shell=True)
