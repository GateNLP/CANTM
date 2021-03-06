import os
import torch
from transformers import *
import nltk
from pathlib import Path
nltk.download('stopwords')
nltk.download('punkt')

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


script_path = os.path.abspath(__file__)
print(script_path)
parent = os.path.dirname(script_path)
parent = os.path.join(parent, 'bert-base-uncased')
print(parent)

model_save_path = os.path.join(parent,'model')
path = Path(model_save_path)
path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(model_save_path)

tokenizer_save_path = os.path.join(parent,'tokenizer')
path = Path(tokenizer_save_path)
path.mkdir(parents=True, exist_ok=True)
tokenizer.save_pretrained(tokenizer_save_path)

