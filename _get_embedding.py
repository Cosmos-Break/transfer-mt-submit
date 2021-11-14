from transformers import BertTokenizer, BertModel
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
emb = model.embeddings.word_embeddings.weight.data
vocab = tokenizer.get_vocab()
v1 = emb[vocab['male']]
v2 = emb[vocab['female']]
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")
emb = model.embeddings.word_embeddings.weight.data
vocab = tokenizer.get_vocab()
v3 = emb[vocab['男']]
v4 = emb[vocab['女']]
torch.norm(v1)
torch.cosine_similarity(v1,v2,0)
res = (v1-v2) - (v3-v4)