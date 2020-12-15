import json
import os
import sys
import random
import math
from pathlib import Path
import copy


def outputChunk(output_dir, chunkIdx, chunks):
    output_path = os.path.join(output_dir, str(chunkIdx))
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(chunks):
        if i == chunkIdx:
            json_path = os.path.join(output_path, 'test.jsonlist')
        else:
            json_path = os.path.join(output_path, 'train.jsonlist')
        with open(json_path, 'a') as jsout:
            for line in item:
                jsout.write(json.dumps(line) + '\n')





input_json = sys.argv[1]
num_fold = int(sys.argv[2])
output_dir = sys.argv[3]



with open(input_json, 'r') as inf:
    data = json.load(inf)
    random.shuffle(data)
    num_data = len(data)
    chunkSize = math.ceil(num_data/num_fold)
    chunks = []
    for i in range(0, len(data), chunkSize):
        chunks.append(copy.deepcopy(data[i:i+chunkSize]))


for idx, eachchunk in enumerate(chunks):
    outputChunk(output_dir, idx, chunks)

