import json
import re
import nltk
import argparse
import copy
import os
import sys
script_path = os.path.abspath(__file__)
print(script_path)
global parent
parent = os.path.dirname(script_path)
print(parent)
gatePython = os.path.join(parent, 'GATEpythonInterface')
sys.path.append(gatePython)
from GateInterface import *
import string
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("inputJson", help="args for test reader")
parser.add_argument("outputJson", help="args for test reader")
args = parser.parse_args()

input_json = args.inputJson
output_json = args.outputJson

with open(input_json, 'r') as fj:
    raw_data = json.load(fj)


doc_id = 0
new_data = []
for item in raw_data:
    doc_id += 1
    item['unique_wv_id'] = str(doc_id)
    new_data.append(copy.deepcopy(item))
    print(doc_id)

with open(output_json, 'w') as fo:
    json.dump(new_data, fo)



