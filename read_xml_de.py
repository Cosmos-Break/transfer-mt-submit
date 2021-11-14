from xml.etree.ElementTree import parse
from nltk.tokenize import sent_tokenize
# doc = parse('IWSLT18.TED.dev2018.eu-en.en.xml')
doc = parse('ted_de.xml')

# Extract and output tags of interest
for item in doc.findall('file/head/transcription'):
    for seg in item.findall('seekvideo'):
        text = seg.text
        if text.startswith('('):
            continue
        # 分句 nltk
        text = sent_tokenize(text)
        for x in text:
            if x.startswith('('):
                continue
            print(x)