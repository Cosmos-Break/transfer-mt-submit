from xml.etree.ElementTree import parse

# doc = parse('IWSLT18.TED.dev2018.eu-en.en.xml')
doc = parse('IWSLT18.TED.tst2018.eu-en.eu.xml')

# Extract and output tags of interest
for item in doc.findall('srcset/doc'):
    for seg in item.findall('seg'):
        text = seg.text
        print(text)