# doc = open('train.tags.eu-en.en')
# doc = open('train.tags.eu-en.eu')
doc = open('sl-en-data/train.tags.sl-en.sl')

# Extract and output tags of interest
for line in doc:
    if line.startswith('<'):
        continue
    print(line.strip())
