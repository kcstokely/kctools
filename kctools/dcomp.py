
keys_i = Counter()
keys_j = Counter()
types  = Counter()
values = Counter()


def item_compare(i, j, base = ''):
    
    
    ki = set(i.keys()).difference(j.keys())
    kj = set(j.keys()).difference(i.keys())
    
    keys_i.update([ base + key for key in  ki ])
    keys_j.update([ base + key for key in  kj ])
    
    keys = set(i.keys()).intersection(j.keys())

    for key in keys:
        
        x = i[key]
        y = j[key]
    
        name = base + key
    
        if not type(x) == type(y):
            types.update([name])
            continue
            
        if type(x) == str:
            if not x == y:
                values.update([name])
                
        if type(x) in [int, float]:
            pct = abs(x-y)/max(x, y)
            if pct < 0.85:
                values.update([name])
                
        if type(x) == dict:
            item_compare(x, y, base = name + '.')

            
            
if __name__ == '__main__':

    with open(X, 'r') as fp:
        x = json.load(fp)
    
    with open(Y, 'r') as fp:
        y = json.load(fp)
        
    x = { i[key]: i for i in x['items'] }
    y = { i[key]: i for i in y['items'] }
    
    item_compare(x, y)
    
    
    
    
    
    
    
    
    
    
    