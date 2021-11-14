We have three child languages, My, Id, Tr.
Here are the my-en, id-en, tr-en parallel corpus.

We train a sub-word tokenizer using SentencePiece for each low-resource language, including My, Id and Tr. The tokenizers are trained on monolingual plain texts which are collected from Wikipedia's dumps(https://dumps.wikimedia.org). The toolkit wikiextractor(https://github.com/attardi/wikiextractor) is utilized to extract plain texts from the semi-structured data.
