########################################################################

import logging, os, re, string
import numpy as np
import pandas as pd
import datetime as dt
from collections import *
from scipy.stats import beta

########################################################################
### OS:

def lsdashr(tdir, absolute = False):
    sdx = 0 if absolute else len(tdir)
    return [ os.path.join(dp, f)[sdx:] for dp, dn, fn in os.walk(tdir) for f in fn ]

def readlines(fname, mode = 'r'):
    with open(fname, mode) as fp:
        return [ line.strip() for line in fp.readlines() ]

########################################################################
### STRINGS:

def lpad(text, x = 2):
    return '\n'.join([ ''.join([' '] * x + list(line)) for line in text.split('\n') ])

###########################

def sbool(text):
    if isinstance(inp, str):
        if text.lower() in ['false', 'no', 'f', 'n', '0']:
            return False
        if text.lower() in ['true', 'yes', 't', 'y', '1']:
            return True
        return None
    return bool(text)

###########################

def html_strip(text):
    return re.sub('<[^<]+?>', '', text)

###########################

def rem_punc(text):
    return ''.join([ y if not y in string.punctuation else '' for y in list(text) ])

def rep_punc(text):
    return ''.join([ y if not y in string.punctuation else ' ' for y in list(text) ])

###########################

def tnow():
    return dt.datetime.strftime(dt.datetime.now(), '%H:%M:%S')

###########################

def endify(n):
    n = str(n)
    if n:
        d = list(n)[-1]
        c = {'1': 'st', '2': 'nd', '3': 'rd'}
        n = n + c.get(d, 'th')
    return n
    
###########################

def humanify(n, l = 2, space = False):
    m = np.abs(n)
    e = int(np.floor(np.log10(m)))
    d = min((e//3)*3, 12)
    c = {0: '', 3: 'k', 6: 'M', 9: 'B', 12: 'T'}[d]
    l = min(e+1, l)
    r = mround(m, np.power(10, max(e-l+1, 0)))/np.power(10, d)
    r = int(r) if (e-d+1 >= l) else r
    return f'{"-" if n < 0 else ""}{r}{" " if space else ""}{c}'

########################################################################
### LISTS:

def only_one(thing):
    return sum(map(bool, thing)) == 1

###########################

def split_into_rows(inlist, m = 5):
    return [ inlist[i:i+m] for i in range(0, len(inlist), m) ]

###########################

def split_into_chunks(inlist, m = 10):
    n = len(inlist)
    r = [ n//m + (i<n%m) for i in range(m) ]
    s = [0] + np.cumsum(r).tolist()
    return  [ inlist[i:j] for i, j in zip(s[:-1], s[1:]) ]

###########################

def flatten(inp):
    return [ j for i in inp for j in flatten(i) ] if isinstance(inp, list) else [ inp ]

###########################

def gram_getter(items, n, strjoin = False):
    grams = list(zip(*[items[i:] for i in range(n)]))
    if strjoin:
        grams = [ ' '.join(gram) for gram in grams ]
    return grams

###########################

def where_in_thing(test, thing):
    
    def get_next_idxs_in_thing(test, thing, idxs = [], already = [], found = False):
        if found or ((thing == test) and (idxs not in already)):
            return idxs, True
        if isinstance(thing, list):
            for i, item in enumerate(thing):
                new_idxs, new_found = get_next_idxs_in_thing(test, item, idxs+[i], already)    
                if new_found:
                    return new_idxs, True
        return idxs, False

    answers = []
    ans, found = get_next_idxs_in_thing(test, thing)
    while(found):
        answers.append(ans)
        ans, found = get_next_idxs_in_thing(test, thing, already = answers)
    
    return answers

########################################################################
### DICTS:

def autovivify(levels = 2, final = int):
    return defaultdict(final) if levels==1 else defaultdict(lambda: autovivify(final, levels-1))

def mortify(thing):
    return { k: mortify(v) for k, v in thing.items() } if isinstance(thing, dict) else thing

########################################################################
### NUMBERS:

def coalesce(*nums):
    a = 0
    for i in nums:
        a = a + i + a*i
    return a    

###########################

def mround(x, m):
    return int(m * round(float(x)/m))

###########################

def normalize(arr):
    return np.nan_to_num(np.divide(arr, np.sum(arr)))

###########################

def is_diagonal(mtx):
    return not np.count_nonzero(mtx - np.diag(np.diagonal(mtx)))

###########################

def kpow(x, p):
    return np.power(np.power(x, 1/p).mean(), p)

def kcos(a, b):
    return np.nan_to_num(np.divide(np.dot(a, b), np.multiply(np.sqrt(np.square(a).sum()), np.sqrt(np.square(b).sum()))), copy=False)

###########################

def bin_entropy(true, pred, eps = 0.0000001):
    return -np.sum( true * np.log(pred+eps) + (1-true) * np.log(1-pred+eps) )

def mod_entropy(true, pred, mod = 2., eps = 0.0000001):
    return -np.sum( true * np.log(pred+eps) + (1-true) * (mod * pred) * np.log(1-pred+eps) )

###########################

def lower_conf_bound(ups, downs, conf = 0.683):
    return beta.ppf((1.-conf)/2., 1+ups, 1+downs)

def upper_conf_bound(ups, downs, conf = 0.683):
    return beta.ppf(1.-(1.-conf)/2., 1+ups, 1+downs)

def conf_bounds(ups, downs, conf = 0.683):
    return (lower_conf_bound(ups, downs, conf), upper_conf_bound(ups, downs, conf))

###########################

def pargsort(arr, n):
    idxs = np.argpartition(arr, n)[:n]
    return idxs[np.argsort(arr[idxs])][:n]

def psort(arr, n):
    return arr[pargsort(arr, n)]

###########################

def rchoice(*args, **kwargs):
    return np.array([], dtype = object) if args and not args[0] else np.random.choice(*args, **kwargs)

########################################################################
### PANDAS:

def rename_dup_df_cols(df, sep = '.'):
    names = pd.Series(df.columns)
    for dup in df.columns.get_duplicates():
        d_mask = df.columns.get_loc(dup)
        if not isinstance(d_mask, int):
            names[d_mask] = [ dup + sep + str(ddx) for ddx in range(d_mask.sum()) ]
    df.columns = names

########################################################################
### LOGGING:

def setup_logger(name = 'kcs', log_file = 'this.log', log_dir = '', mode = 'a', level = 'info'):
    fmt = '%(asctime)s - %(levelname)4s: %(message)s'
    datefmt = '%Y-%m-%d - %H:%M:%S'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    level = getattr(logging, level.upper()) if isinstance(level, str) else level
    logger = logging.getLogger(name)
    logger.setLevel(level)
    f_handler = logging.FileHandler(os.path.join(log_dir, log_file), mode)
    f_handler.setLevel(level)
    f_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(f_handler)
    return logger

########################################################################



















