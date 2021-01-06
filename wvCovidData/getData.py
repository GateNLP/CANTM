import wget
import os
import zipfile

script_path = os.path.abspath(__file__)
print(script_path)
parent = os.path.dirname(script_path)
print(parent)

data_url = 'https://gate.ac.uk/g8/page/show/2/gatewiki/cow/covid19catedata/covidCateData.zip'
wget.download(data_url, parent)

zip_path = os.path.join(parent, 'covidCateData.zip')

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(parent)

