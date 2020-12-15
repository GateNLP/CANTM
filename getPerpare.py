import os
import torch
from transformers import *
import nltk
from pathlib import Path
nltk.download('stopwords')
nltk.download('punkt')

script_path = os.path.abspath(__file__)
print(script_path)
parent = os.path.dirname(script_path)
parent = os.path.join(parent, 'bert-base-uncased')
print(parent)

model_save_path = os.path.join(parent,'model')
path = Path(model_save_path)
path.mkdir(parents=True, exist_ok=True)

model = BertModel.from_pretrained('bert-base-uncased')
model.save_pretrained(model_save_path)
