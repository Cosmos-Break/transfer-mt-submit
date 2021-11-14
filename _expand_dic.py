for line in open('en-eu.dic'):
    print(line.strip())
    a = line.strip().split()
    line = '▁' + a[0] + ' ' + '▁' + a[1]
    print(line)