# CANTM
Implementation of Classification Aware Neural Topic Model (CANTM) and application for misinformation category classification
Please cite the following paper in your publication involving this work:
Xingyi Song, Johann Petrak, Ye Jiang, Iknoor Singh, Diana Maynard, Kalina Bontcheva, Classification Aware Neural Topic Model and its Application on a New COVID-19 Disinformation Corpus, In arXiv:2006.03354, 2020.


## Install
* Requirement:
  * miniconda: https://docs.conda.io/en/latest/miniconda.html
* Install conda environment: conda env create -f environmentGPU.yml
* Active conda environment: conda activate wvmisinfoGPU
* Download NLTK and BERT models: python getPerpare.py

## Run Experiments:
### Covid Exp
* Get COVID data: 
```
cd wvCovidData
bash getCovidData.sh
```
* Run CANTM: bash runCANTM_covid.sh
* Run BERT: bash runBert_covid.sh
* Run NVDM: runNVDM_covid.sh
* Run NVDM_bert: runNVDM_bert_covid.sh
* Run LDA: runLDA_covid.sh


### Update topics with unlabelled data:
* Update classification-aware topic on COVID (note require `bash runCANTM_covid.sh' first), the latest unlabelled data not included please contact author. for the data: bash updateCovid.sh 

### Run Covid data on SCHOLAR:
* Perpare data for SCHOLAR: bash outputForScholar.sh 
* The Scholar ready data format (splited in 5 folds) will be in 'testScholarfold'
* Please follow the instruction in Scholar to run the experiment
* Patched version of 'run_scholar.py' in Scholar_patch/, that output f1 score. (Copy to original Scholar folder to use)









