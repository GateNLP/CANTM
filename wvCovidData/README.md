# covid_annotation

This is code is only for EMNLP review, please do not distribute. 

## Structure:

* README.md: This file
* source_data: The folder contains the unannotated source data
* annotated_data: The folder contains all annotated data
* mergedData: The folder contains the merged data
  * merged_all.json/tsv: Is merged data from all annotated files (exclude SocAlrm)
    * 1480 annotated samples
    * overall agreement:  0.5183150183150184
    * overall kappa:  0.4701465201465202
    * total pair compareed:  1092

  * merged_clean.json/tsv: Is the selected merged data
    * 1293 annotated samples
    * overall agreement:  0.7247706422018348
    * overall kappa:  0.6972477064220183
    * total pair compareed:  436

  * merged_clean_ml.json/tsv: Merged Prot class to PubPrep
    * 1293 annotated samples
    * overall agreement:  0.7270642201834863
    * overall kappa:  0.6967380224260958
    * total pair compareed:  436
    * class statistics: {'PubAuthAction': 323, 'CommSpread': 262, 'PubPrep': 82, 'PromActs': 302, 'GenMedAdv': 215, 'VirTrans': 95, 'Vacc': 104, 'None': 60, 'Consp': 119, 'VirOrgn': 80} 

* mergeAnnos.py: Python script for data merge and agreement calcucation
* WvLibs: Folder contains WeVerify Covid data reader library
* run.sh: The bash script generate merged_all.json/tsv
* run_clean.sh: The bash script generate merged_clean.json/tsv


## mergeAnnos.py:
* Requirement: Python 3
* Useage: `python mergeAnnos.py raw_json_dir annoed_json_dir merged_json --output2csv merged_tsv [options]`
  * raw_json_dir: The folder contains the unannotated source data (source_data in this folder)
  * annoed_json_dir: The folder contains all annotated data (annotated_data in this folder)
  * merged_json: file path to merged json file
  * merged_tsv: file path to merged tsv file (This is required for accurate calculate kappa)
* Options inlcude:
```
optional arguments:
  -h, --help            show this help message and exit
  --ignoreLabel IGNORELABEL
                        ignore label, splited by using ,
  --ignoreUser IGNOREUSER
                        ignore user, splited by using ,
  --min_anno_filter MIN_ANNO_FILTER
                        min annotation frequence
  --min_conf_filter MIN_CONF_FILTER
                        min confident
  --output2csv OUTPUT2CSV
                        output to csv
  --transfer_label TRANSFER_LABEL
                        trasfer label to another category, in format:
                        orilabel1:tranlabel1,orilabel2:tranlabel2
  --cal_agreement       calculate annotation agreement
  --logging_level LOGGING_LEVEL
                        logging level, default warning, other option inlcude
                        info and debug
  --user_conf USER_CONF
                        User level confident cutoff threshold, in format:
                        user1:thres1,user2:thres2
  --set_reverse         reverse the selection condition, to check what
                        discared
```
  * Please check run_clean.sh for example

## Data Qulity Check:
Please check this sheet for individual quality check (2 tabs inlcuded in the sheet):
https://docs.google.com/spreadsheets/d/1vUSIP4lAQFPNVYCtCyDY_P2Hx3_PmPNQ8Qdzga-SLIc/edit#gid=0

To balance the quality and number of data (merged_clean.json/tsv) following selection strategy was used:
* User 04 and 58 are ignored
* Global confidence threshold set to 6 
  * anything annotation below or equal confident 6 are ignored
  * except the individual condifence was set in user_conf
* Individual confidence
  * 23:7
  * 03:7
  * 22:2
  * 00:1 
  * 12:5
  * 13:7
  * 07:6
  * 25:5
  * 16:5
  * 05:4
  * 06:5
  * 08:5

## Merge strategy:
  The merged (agreement resolve) label is stored in 'selected_label' field in the json file, and based on following rules:
  * If single annotated --> selected the label
  * else if all agree on the annotation --> select the label
  * else if have majority agreement on the label --> select the majority agreed label
  * else if have the highest summed confidence label --> select the label with highest summed confidence
  * else select the one with the highest confidence



