import sys
L2 = sys.argv[1]
def print_emb(filename):
    with open(filename) as f:
        n, emb_size = f.readline().strip().split()
        n = int(n)
        emb_size = int(emb_size)
        for line in f:
            line = line.strip().split()[1:]
            assert len(line) == 512
            print(' '.join(line))

print_emb('opus-mt-de-en/embedding')
print_emb(L2)
# opus-mt-de-en/embedding 58101
# vectors-eu.txt 50099