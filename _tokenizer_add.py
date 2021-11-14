from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTokenizedInput
from tokenizers import CharBPETokenizer
from tokenizers.pre_tokenizers import Whitespace

L1, L2 = [], []
for line in open('berteus-base-cased/vocab.txt'):
    line = line.strip()
    if line.startswith('##'):
        line = '▁' + line[2:]
    L1.append(line)
for line in open('opus-mt-de-en/vocab.txt'):
    line = line.strip()
    L2.append(line)

L1 = set(L1[5:])
L2 = set(L2)
cnt = 0
tokens = set()
for token in L2:
    if token in L1:
        token = token.lower()
        tokens.add(token)

model_checkpoint = "opus-mt-de-en"
pretrained_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(pretrained_tokenizer)

model_inputs = pretrained_tokenizer('There have been three themes running through the conference which are relevant to what I want to talk about!')
print(pretrained_tokenizer.convert_ids_to_tokens(model_inputs['input_ids']))

pretrained_tokenizer_my = CharBPETokenizer(vocab='my-tokenizer/vocab.json', merges='my-tokenizer/merges.txt')
model_inputs = pretrained_tokenizer_my.encode('ရန့်ဝစ်(ခ်) ကို ပိတ်ထားခဲ့ ပြီး ၂လ အထိ ကြာကြာ ဆက်လက်ထိမ်းသိမ်းထား ရန် မျှော်လင့်ပါတယ် ။!')
print(model_inputs.tokens)

model_checkpoint = "berteus-base-cased"
pretrained_tokenizer_eu = AutoTokenizer.from_pretrained(model_checkpoint)

cnt = 0
for token in L1:
    if token not in L2:
        pretrained_tokenizer.add_tokens(token)
        cnt += 1
print(cnt)
# pretrained_tokenizer.save_pretrained('opus-mt-de-en-tokenizer-added')
print(123)

pretrained_tokenizer_new = AutoTokenizer.from_pretrained('opus-mt-de-en-tokenizer-added')

# ret = pretrained_tokenizer_new('Egun on. Zer moduz? Izugarria izan da, ezta?')
pretrained_tokenizer_eu('Egun on. Zer moduz? Izugarria izan da, ezta?')
print(123)
pretrained_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
pretrained_tokenizer.add_tokens(['zelanbait'])
