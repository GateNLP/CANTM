python evaluation.py wvCovidData/mergedData/merged_clean_ml.json --configFile configFiles/sampleConfig.cfg --corpusType wvmisinfoScholar --export_json covidExport.json
rm -r testScholarfold
python raw2folds.py covidExport.json 5 testScholarfold
