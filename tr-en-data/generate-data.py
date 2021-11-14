from xml.etree.ElementTree import parse
from bs4 import BeautifulSoup
# doc = parse('IWSLT18.TED.dev2018.eu-en.en.xml')
for line in open('tr-en-data/newstest2017-tren-src.tr.sgm'):
    if line.startswith('<seg id='):
        line = BeautifulSoup(line, "html.parser").text
        print(line.strip())
