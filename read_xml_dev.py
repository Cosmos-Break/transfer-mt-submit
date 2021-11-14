from xml.etree.ElementTree import parse

# doc = parse('IWSLT18.TED.dev2018.eu-en.en.xml')
# doc = parse('IWSLT18.TED.dev2018.eu-en.eu.xml')
# doc = parse('sl-en-data/IWSLT14.TED.tst2012.sl-en.sl.xml')
doc = parse('sl-en-data/IWSLT14.TED.tst2012.sl-en.en.xml')

# Extract and output tags of interest
# for item in doc.findall('srcset/doc'):
for item in doc.findall('refset/doc'):
    for seg in item.findall('seg'):
        text = seg.text
        print(text.strip())