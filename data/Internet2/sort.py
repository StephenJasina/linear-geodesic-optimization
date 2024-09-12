with open('links.txt', 'r') as f:
    lines = sorted(list(f.readlines()))

with open('links_new.txt', 'w') as f:
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        f.write(line)
        f.write('\n')
