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

label_dict = {
        'no_evidence':['NO EVIDENCE', 'No evidence', 'No Evidence', 'no evidence', 'Unproven', 'Unverified'],
        'misleading': ['misleading', 'Misleading', 'MISLEADING', 'mislEADING', 'MiSLEADING'],
        'false': ['Pants on Fire!', 'False', 'FALSE', 'Not true', 'false and misleading', 'false', 'PANTS ON FIRE', 'Fake news', 'Misleading/False', 'Fake', 'Incorrect'],
        'partial_false': ['Partially correct', 'mostly false', 'HALF TRUTH', 'HALF TRUE', 'partly false', 'Mostly true', 'Partly true', 'Mixed', 'half true', 'True but', 'MOSTLY FALSE', 'Partially false', 'PARTLY FALSE', 'Partially true', 'MOSTLY TRUE', 'Partly false', 'Mostly False', 'Partly False', 'PARTLY TRUE', 'Mostly True', 'Half True', 'Mostly false'],
        }


with open(input_json, 'r') as fj:
    raw_data = json.load(fj)


doc_id = 0
new_data = []
for item in raw_data:
    item_keys = item.keys()
    print(item_keys)
    label = item['Label']
    label_group = 'other'
    for each_label_group in label_dict:
        if label in label_dict[each_label_group]:
            label_group = each_label_group
            break
    doc_id += 1
    item['ori_label'] = label
    item['Label'] = label_group
    new_data.append(copy.deepcopy(item))
    print(doc_id)

with open(output_json, 'w') as fo:
    json.dump(new_data, fo)



