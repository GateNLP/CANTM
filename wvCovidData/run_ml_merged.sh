#! /bin/bash

python mergeAnnos.py source_data/ annotated_data/ mergedData/merged_clean_ml.json --ignoreLabel SocAlrm --ignoreUser 04,58 --min_anno_filter 1 --output2csv mergedData/merged_clean_ml.tsv --min_conf_filter 6 --cal_agreement --transfer_label :None,Prot:PubPrep --user_conf 23:7,03:7,22:2,00:1,12:5,13:7,07:6,25:5,16:5,05:4,06:5,08:5
