import sys
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.webdriver.firefox.options import Options


def download_source(url):
    options = Options()
    options.headless = True
    driver = webdriver.Firefox(options=options)
    driver.get(url)
    html = driver.page_source
    driver.close()

    soup = BeautifulSoup(html, 'html.parser')
    factcheck_Org = soup.find('p', attrs={'class':'entry-content__text entry-content__text--org'}).get_text()[17:]
    date = soup.find('p', attrs={'class':'entry-content__text entry-content__text--topinfo'}).get_text()[0:10]
    country = soup.find('p', attrs={'class':'entry-content__text entry-content__text--topinfo'}).get_text()[13:]
    ct = soup.find('h1', attrs={'class':'entry-title'})
    for tag in ct.find_all('span'):
        tag.replaceWith('')

    claim = ct.get_text()
    explain = soup.find('p', attrs={'class':'entry-content__text entry-content__text--explanation'}).get_text()[13:]
    print(claim)
    print(explain)
    print(country)
    print(date)
    print(factcheck_Org)
    return claim,explain,country,date,factcheck_Org



input_json = sys.argv[1]
output_json = sys.argv[2]


with open(input_json, 'r') as fin:
    data = json.load(fin)

for i, each_data in enumerate(data):
    claim,explain,country,date,factcheck_Org = download_source(each_data['Link'])
    each_data['Claim'] = claim
    each_data['Explaination'] = explain
    each_data['Country'] = country
    each_data['Date'] = date
    each_data['Factcheck_Org'] = factcheck_Org
    print(i, len(data))
    #break

with open(output_json, 'w') as fo:
    json.dump(data, fo)
