#! /bin/bash

python mergeAnnos.py source_data/ annotated_data/ mergedData/merged_all.json --ignoreLabel SocAlrm --output2csv mergedData/merged_all.tsv --transfer_label :None --cal_agreement
