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
from tokenizers import ByteLevelBPETokenizer
from transformers.trainer_callback import TrainerControl
from torch.utils.data import DataLoader
import time
import numpy as np

model_checkpoint = "opus-mt-de-en"
pretrained_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(pretrained_tokenizer)

pretrained_tokenizer_eu = ByteLevelBPETokenizer('eu-tokenizer/vocab.json', 'eu-tokenizer/merges.txt')
print(pretrained_tokenizer_eu)

bos_token = '<s>'
unk_token = '<unk>'
eos_token = '</s>'
pad_token = '<pad>'
pad_token_id = pretrained_tokenizer_eu.token_to_id(pad_token)
eos_token_id = pretrained_tokenizer_eu.token_to_id(eos_token)


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
    for encoding in pretrained_tokenizer_eu.encode_batch([sources]):
        input_ids = encoding.ids
        attention_mask = encoding.attention_mask
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

tokenized_dataset_dict = dataset_dict.map(batch_tokenize_fn)
print(tokenized_dataset_dict)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_model = torch.load('opus-mt-de-en/pytorch_model_concat_emb.bin').to(device)
# pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)
print('# of parameters: ', pretrained_model.num_parameters())


batch_size = 4
args = Seq2SeqTrainingArguments(
    output_dir="test-translation",
    evaluation_strategy="steps",
    learning_rate=0.0005,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    load_best_model_at_end=True,
    predict_with_generate=True,
    remove_unused_columns=True,
    fp16=False
)




data_collator = Seq2SeqDataCollator(max_source_length, pad_token_id)

callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

trainer = Seq2SeqTrainer(
    pretrained_model,
    args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset_dict["train"],
    eval_dataset=tokenized_dataset_dict["val"],
    callbacks=callbacks
)


dataloader_train = trainer.get_train_dataloader()
batch = next(iter(dataloader_train))
# trainer_output = trainer.train()
print(123)







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

data_collator = Seq2SeqDataCollator(max_source_length, pad_token_id, pad_token_id)

data_loader = DataLoader(tokenized_dataset_dict['test'], collate_fn=data_collator, batch_size=4)
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
    # for target, prediction in zip(targets, predictions):
    #     print("target:", target)
    #     print("prediction:", prediction)
with open('target', 'w') as f:
    for line in target_list:
        f.write(line + '\n')
with open('prediction', 'w') as f:
    for line in prediction_list:
        f.write(line + '\n')


# def generate_translation_testset(model, tokenizer, examples):
#     for example in examples:
#         """print out the source, target and predicted raw text."""
#         source = example[source_lang]
#         target = example[target_lang]
#         input_ids = example['input_ids']
#         input_ids = torch.LongTensor(input_ids).view(1, -1).to(model.device)
#         # print('input_ids: ', input_ids)
#         generated_ids = model.generate(input_ids)
#         # print('generated_ids: ', generated_ids)
#         prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
#         # print('source: ', source)
#         print('target: ', target)
#         print('prediction: ', prediction)

# generate_translation_testset(pretrained_model, pretrained_tokenizer, tokenized_dataset_dict['test'])


example = tokenized_dataset_dict['train'][0]
generate_translation(pretrained_model, pretrained_tokenizer, example)

