import os

dirname = os.path.dirname(__file__)

names = {}
path = os.path.join(dirname, 'names.txt')
with open(path, 'r') as fp:
    for line in fp.readlines():
        line = line.strip()
        if (not line) or line[0] == '#':
            continue
        name, notes = line.strip().split(':')
        notes = tuple( int(i.strip()) for i in notes.split(',') )
        names[name] = notes
        names[notes] = name

zeitler = {}
path = os.path.join(dirname, 'scale-list.csv')
with open(path, 'r') as fp:
    fp.readline()
    for line in fp.readlines():
        line = line.strip()
        if line:
            _, _, name, tones = line.split(',')
            tones = list(tones.strip())
            tones = map(int, tones)
            notes = [ 0 ]
            for tone in tones:
                notes += [ tone + notes[-1] ]
            notes = notes[:-1]
            zeitler[name.lower()] = tuple(notes)
            zeitler[tuple(notes)] = name.lower()
