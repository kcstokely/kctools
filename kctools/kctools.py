########################################################################

import logging, math, os, re, string
import datetime as dt

from copy        import deepcopy
from collections import *

import numpy  as np
import pandas as pd

from scipy.stats import beta
from matplotlib  import pyplot as plt

########################################################################
### FILES:

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
    m = math.abs(n)
    e = int(math.log10(m))
    d = min((e//3)*3, 12)
    c = {0: '', 3: 'k', 6: 'M', 9: 'B', 12: 'T'}[d]
    l = min(e+1, l)
    r = mround(m, 10**(max(e-l+1, 0)))/(10**d)
    r = int(r) if (e-d+1 >= l) else r
    return f'{"-" if n < 0 else ""}{r}{" " if space else ""}{c}'

########################################################################
### LISTS:

def only_one(inlist):
    return sum(map(bool, inlist)) == 1

###########################

def split_into_rows(inlist, m = 5):
    return [ inlist[i:i+m] for i in range(0, len(inlist), m) ]

###########################

def split_into_chunks(inlist, m = 10):
    n = len(inlist)
    r = [ n//m + (i<n%m) for i in range(m) ]
    s = [0] + [ sum(r[:x]) for x in range(m) ]
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
    '''thanks bert'''
    return defaultdict(final) if levels==1 else defaultdict(lambda: autovivify(levels-1, final))

def mortify(inp):
    return { k: mortify(v) for k, v in inp.items() } if isinstance(inp, dict) else inp

def dict_update(A, B, new = False):
    if new:
        A = deepcopy(A)
    for key, value in B.items():
        if key in A and isinstance(value, dict):
            A[key] = dict_update(A[key], value)
        else:
            A[key] = value
    return A

########################################################################
### LOGGING:

def setup_logger(
        name,
        path  = '',
        mode  = 'a',
        level = 'debug'
    ):
    
    fmt     = '%(asctime)s - %(levelname)4s: %(message)s'
    datefmt = '%Y-%m-%d - %H:%M:%S'
    level   = getattr(logging, level.upper()) if isinstance(level, str) else level

    logger = logging.getLogger(name)
    logger.setLevel(level)

    s_handler = logging.StreamHandler()
    s_handler.setLevel(level)
    s_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(s_handler)
    
    if path:
        os.makedirs(os.path.dirname(path), exist_ok = True)
        f_handler = logging.FileHandler(path, mode)
        f_handler.setLevel(level)
        f_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(f_handler)
    
    return logger

########################################################################
### NUMBERS: 

def mround(x, m):
    return int(m * round(float(x)/m))

def coalesce(*these):
    y = 0
    for x in these:
        y = y + x + y*x
    return y

########################################################################
### NUMPY:

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

def pargsort(arr, n):
    idxs = np.argpartition(arr, n)[:n]
    return idxs[np.argsort(arr[idxs])][:n]

def psort(arr, n):
    return arr[pargsort(arr, n)]

###########################

def rchoice(*args, **kwargs):
    try:
        return np.random.choice(*args, **kwargs)
    except ValueError:
        size = kwargs.get('size') or args[1] or 1
        if size < 2:
            return None
        else:
            return np.array([], dtype = object)

###########################
### SCIPY

def lower_conf_bound(ups, downs, conf = 0.683):
    return beta.ppf((1.-conf)/2., 1+ups, 1+downs)

def upper_conf_bound(ups, downs, conf = 0.683):
    return beta.ppf(1.-(1.-conf)/2., 1+ups, 1+downs)

def conf_bounds(ups, downs, conf = 0.683):
    return (lower_conf_bound(ups, downs, conf), upper_conf_bound(ups, downs, conf))

########################################################################
### PANDAS:

def rename_dup_df_cols(df, sep = '.'):
    names = pd.Series(df.columns)
    for dup in df.columns.get_duplicates():
        d_mask = df.columns.get_loc(dup)
        names[d_mask] = [ f'{dup}{sep}{ddx}' for ddx in range(d_mask.sum()) ]
    df.columns = names

########################################################################
### MATPLOTLIB:

def make_heatmap(
            data,
            xlabels  = None,
            ylabels  = None,
            annotate = True,
            title    = None,
            subtitle = None,
            xlabel   = None,
            ylabel   = None,
            norm     = None,
            cmap     = 'copper',
            tight    = True,
            figsize  = (9, 9),
            fpath    = None,
            show     = None
    ):

    data = np.array(data)
    assert len(data.shape) == 2

    if norm:
        copy = data.copy()
        for i in range(num):
            for j in range(num):
                data[i, j] = data[i, j] / np.min([copy[i, i], copy[j, j]])
        if norm == 'pct':
            data = (data * 100).astype(int)

    fmt = '.2f'
    if all([ (not x%1) for x in data.flatten() ]):
        fmt = 'd' 
        data = data.astype(int)

    if not xlabels:
        xlabels = [ str(i) for i in range(len(data[0]))]
    if not ylabels:
        ylabels = [ str(i) for i in range(len(data[1]))]

    #########

    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(data, cmap = 'copper', origin='lower', interpolation='nearest')
  
    if title:
        fig.suptitle(title, fontsize=12)
    if subtitle:
        ax.set_title(subtitle)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
  
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))

    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
  
    if annotate:
        for idx in range(len(data[0])):
            for jdx in range(len(data[1])):
                value = f'{data[idx, jdx]:{fmt}}'
                if value[0] == '0':
                    value = value[1:]
            ax.text(jdx, idx, value, ha='center', va='center', color='w')
 
    if tight:
        fig.tight_layout()
  
    if fpath:
        plt.savefig(fpath)
    if show or (show is None and not fpath):
        plt.show()
        
    return fig, ax

########################################################################



















