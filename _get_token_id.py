import os
from datasets import load_dataset
from transformers import AutoTokenizer

model_checkpoint = "opus-mt-de-en"
pretrained_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(pretrained_tokenizer)

model_checkpoint = "berteus-base-cased"
pretrained_tokenizer_eu = AutoTokenizer.from_pretrained(model_checkpoint)
pretrained_tokenizer_eu('123', add_special_tokens=False)
print(1)
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
print(123)

max_source_length = 128
max_target_length = 128
source_lang = "eu"
target_lang = "en"

# orig_emb_size
orig_emb_size = 58101
def batch_tokenize_fn(examples):
    """
    Generate the input_ids and labels field for huggingface dataset/dataset dict.
    
    Truncation is enabled, so we cap the sentence to the max length, padding will be done later
    in a data collator, so pad examples to the longest length in the batch and not the whole dataset.
    """
    sources = examples[source_lang]
    targets = examples[target_lang]
    model_inputs = pretrained_tokenizer_eu(sources, max_length=max_source_length, truncation=True, add_special_tokens=False)
    model_inputs.pop('token_type_ids')
    n = len(model_inputs['input_ids'])
    for i in range(n):
        m = len(model_inputs['input_ids'][i])
        for j in range(m):
            model_inputs['input_ids'][i][j] += orig_emb_size
        model_inputs['input_ids'][i].append(0)
    # pretrained_tokenizer_eu.convert_ids_to_tokens(model_inputs['input_ids'])

    # setup the tokenizer for targets,
    # huggingface expects the target tokenized ids to be stored in the labels field
    with pretrained_tokenizer.as_target_tokenizer():
        labels = pretrained_tokenizer(targets, max_length=max_target_length, truncation=True)
    # pretrained_tokenizer.convert_ids_to_tokens(labels['input_ids'])

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset_dict = dataset_dict.map(batch_tokenize_fn, batched=True)
print(tokenized_dataset_dict)