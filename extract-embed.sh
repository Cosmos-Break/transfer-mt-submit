#!/bin/bash

# Usage: ./extract-embed.sh [PARAM_FILE] [VOCAB_FILE] [source|target]

# Inputs
# PARAM_FILE=${1}
# VOCAB_JSON=${2}
# SIDE=${3}
PARAM_FILE=transformer_en_de_512_WMT2014-e25287c5.params
VOCAB_JSON=WMT2014_src-230ebb81.vocab
SIDE=src

# Outputs
# EMBED_TXT=${PARAM_FILE}.${SIDE}_embed_weight
EMBED_TXT=${PARAM_FILE}.${SIDE}_embed.0.weight

VOCAB_TXT=${VOCAB_JSON::-5}.txt
OUTPUT_FILE=${EMBED_TXT}.vec

# Extract embed
echo "Extracting ${SIDE} embed..."
python -m sockeye.extract_parameters --names ${SIDE}_embed.0.weight --text-output --output ${PARAM_FILE} ${PARAM_FILE}
# Convert vocab file from json to txt
# 1. Counts dropped
# 2. Backslash removed from double quote ("), backslash (\)
# WARNING: Unicode combining class characters are not properly printed in the screen (look at the end of the previous line!)
echo "Converting vocab..."
# cat ${VOCAB_JSON} | sed '1d;$d;s/\": .\+$//;s/\"//;s/ //g' | sed 's/^\\//' > ${VOCAB_TXT}
python convert_vocab_to_txt.py ${VOCAB_TXT}

# Paste vocab and embed
echo "Writing .vec format..."
paste -d' ' ${VOCAB_TXT} ${EMBED_TXT} > ${OUTPUT_FILE}
sed -i "1s/^/$(wc -l ${VOCAB_TXT} | cut -d' ' -f1) $(head -n1 ${EMBED_TXT} | wc -w)\n/" ${OUTPUT_FILE}

# Finish
echo "Cleaning up..."
rm ${EMBED_TXT}
