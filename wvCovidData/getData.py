import wget
import os
import zipfile

script_path = os.path.abspath(__file__)
print(script_path)
parent = os.path.dirname(script_path) 
print(parent)

data_url = 'http://staffwww.dcs.shef.ac.uk/people/X.Song/weVerifyHackathonData/wvHackathonData.zip'
wget.download(data_url, parent)

zip_path = os.path.join(parent, 'wvHackathonData.zip')

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(parent)


