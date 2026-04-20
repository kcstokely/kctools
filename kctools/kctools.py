import datetime as dt
import logging, math, os, re, string, sys

from copy        import deepcopy
from collections import defaultdict
from importlib   import util

is_np  = util.find_spec('numpy') is not None
is_pd  = util.find_spec('pandas') is not None
is_tq  = util.find_spec('tqdm') is not None
is_sci = util.find_spec('scipy') is not None
is_mpl = util.find_spec('matplotlib') is not None

if is_np:
    import numpy  as np
if is_pd:    
    import pandas as pd
if is_sci:    
    from scipy.stats import beta
if is_mpl:
    from matplotlib import pyplot as plt

def np_check(method):
    def wrapped(*args, **kwargs):
        if is_np:
            return method(*args, **kwargs)
        else:
            raise Exception('Please install numpy.')
    return wrapped

def pd_check(method):
    def wrapped(*args, **kwargs):
        if is_pd:
            return method(*args, **kwargs)
        else:
            raise Exception('Please install pandas.')
    return wrapped

def npd_check(method):
    def wrapped(*args, **kwargs):
        if is_np:
            return method(*args, **kwargs)
        else:
            raise Exception('Please install numpy, and maybe pandas.')
    return wrapped

def sci_check(method):
    def wrapped(*args, **kwargs):
        if is_sci:
            return method(*args, **kwargs)
        else:
            raise Exception('Please install scipy.')
    return wrapped
     
def mpl_check(method):
    def wrapped(*args, **kwargs):
        if is_mpl:
            return method(*args, **kwargs)
        else:
            raise Exception('Please install matplotlib.')
    return wrapped

########################################################################
########################################################################
### FILES:

def lsdashr(path, absolute = False):
    sdx = 0 if absolute else len(path) + len(os.sep)
    return [ os.path.join(dp, f)[sdx:] for dp, dn, fn in os.walk(path) for f in fn ]

def readlines(path):
    with open(path, 'r') as fp:
        return [ line.strip() for line in fp.readlines() ]

########################################################################
########################################################################
### STRINGS:

def lpad(text, x = 2):
    return '\n'.join([ ''.join([' '] * x + list(line)) for line in text.split('\n') ])

###########################

def sbool(text):
    if isinstance(text, str):
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
    return ''.join([ y if not y in string.punctuation else '' for y in text ])

def rep_punc(text):
    return ''.join([ y if not y in string.punctuation else ' ' for y in text ])

###########################

def tnow():
    return dt.datetime.strftime(dt.datetime.now(), '%H:%M:%S')

###########################

def endify(n):
    n = str(n)
    if n:
        neg = False
        if n[0] == '-':
            n = n[1:]
            neg = True
        if len(n) >= 2 and n[-2] == '1':
            return n + 'th'
        c = {'1': 'st', '2': 'nd', '3': 'rd'}
        n = n + c.get(n[-1], 'th')
        if neg:
            n = '-' + n
    return n
    
###########################

def humanify(n, l = 2, space = False):
    n = float(n)
    m = abs(n)
    if m == 0:
        return f'{"-" if n < 0 else ""}0{" " if space else ""}'
    e = int(math.log10(m))
    d = min((e//3)*3, 12)
    c = {0: '', 3: 'k', 6: 'M', 9: 'B', 12: 'T'}[d]
    l = min(e+1, l)
    r = mround(m, 10**(max(e-l+1, 0)))/(10**d)
    r = int(r) if (e-d+1 >= l) else r
    return f'{"-" if n < 0 else ""}{r}{" " if space else ""}{c}'

########################################################################
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
    s = [ sum(r[:x]) for x in range(m+1) ]
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
########################################################################
### DICTS:

def autovivify():
    '''thanks eu'''
    return defaultdict(lambda: autovivify())

def vivify(levels = 2, final = int):
    '''thanks bert'''
    return defaultdict(final) if levels==1 else defaultdict(lambda: vivify(levels-1, final))

def mortify(inp):
    return { k: mortify(v) for k, v in inp.items() } if isinstance(inp, dict) else inp

def invert(x):
    y = defaultdict(list)
    for k, vlist in x.items():
        for v in vlist:
            y[v].append(k)
    return dict(y)

def dict_update(A, B, inplace = False):
    if not inplace:
        A = deepcopy(A)
    for key, value in B.items():
        if key in A and isinstance(value, dict):
            A[key] = dict_update(A[key], value)
        else:
            A[key] = value
    return A

def dict_compare(A, B):
    """Compare two dictionaries and return basic differences."""
    differences = {}
    
    # Keys in A not in B or with different values
    for key in A:
        if key not in B:
            differences[key] = {'in_A': A[key], 'in_B': None}
        elif A[key] != B[key]:
            differences[key] = {'in_A': A[key], 'in_B': B[key]}
    
    # Keys in B not in A
    for key in B:
        if key not in A:
            differences[key] = {'in_A': None, 'in_B': B[key]}
    
    return differences

########################################################################
########################################################################
### LOGGING:

# add color formatting?

def setup_logger(
        name    = 'kctools',
        path    = '',
        mode    = 'a',
        level   = 'debug',
        f_level = 'debug',
        console = True,
        c_level = 'debug'
    ):

    fmt     = '%(asctime)s  %(levelname).4s: %(message)s'
    datefmt = '%Y.%m.%d  %H:%M:%S'
    level = getattr(logging, level.upper()) if isinstance(level, str) else level
    c_level = getattr(logging, c_level.upper()) if isinstance(c_level, str) else c_level
    f_level = getattr(logging, f_level.upper()) if isinstance(f_level, str) else f_level

    logger = logging.getLogger(name)
    logger.setLevel(level)

    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)
    
    if console:
        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setLevel(c_level)
        s_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(s_handler)
    
    if path:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok = True)
        f_handler = logging.FileHandler(path, mode)
        f_handler.setLevel(f_level)
        f_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(f_handler)
    
    return logger

########################################################################
########################################################################
### NUMBERS: 

def mround(x, m):
    if not m:
        return x
    y = m * round(float(x)/abs(m))
    return int(y) if m >= 1 else y

def compare(x, y, other = None):
    i = x > other if other is not None else True
    j = y > other if other is not None else True
    if i ^ j:
        return x if i else y
    else:
        return x if x > y else y

def coalesce(*values):
    y = 0
    for x in values:
        y = y + x + y*x
    return y

########################################################################
########################################################################
### NUMPY:

@np_check
def normalize(arr):
    return np.nan_to_num(np.divide(arr, np.sum(arr)))

###########################

@np_check
def is_diagonal(mtx):
    return not np.count_nonzero(mtx - np.diag(np.diagonal(mtx)))

###########################

@np_check
def kpow(x, p):
    return np.power(np.power(x, 1/p).mean(), p)

@np_check
def kcos(a, b):
    return np.nan_to_num(np.divide(np.dot(a, b), np.multiply(np.sqrt(np.square(a).sum()), np.sqrt(np.square(b).sum()))), copy=False)

###########################

@np_check
def bin_entropy(true, pred, eps = 0.0000001):
    return -np.sum( true * np.log(pred+eps) + (1-true) * np.log(1-pred+eps) )

@np_check
def mod_entropy(true, pred, mod = 2., eps = 0.0000001):
    return -np.sum( true * np.log(pred+eps) + (1-true) * (mod * pred) * np.log(1-pred+eps) )

###########################

@np_check
def pargsort(arr, n):
    idxs = np.argpartition(arr, n)[:n]
    return idxs[np.argsort(arr[idxs])][:n]

@np_check
def psort(arr, n):
    return arr[pargsort(arr, n)]

###########################

@np_check
def rchoice(*args, **kwargs):
    try:
        return np.random.choice(*args, **kwargs)
    except ValueError:
        size = kwargs.get('size')
        if not size:
            if len(args) > 1:
                size = args[1]
            else:
                size = 1
        if size < 2:
            return None
        else:
            return np.array([], dtype = object)

########################################################################
########################################################################
### SCIPY:

@sci_check
def lower_conf_bound(ups, downs, conf = 0.683):
    return beta.ppf((1.-conf)/2., 1+ups, 1+downs)

@sci_check
def upper_conf_bound(ups, downs, conf = 0.683):
    return beta.ppf(1.-(1.-conf)/2., 1+ups, 1+downs)

@sci_check
def conf_bounds(ups, downs, conf = 0.683):
    return (lower_conf_bound(ups, downs, conf), upper_conf_bound(ups, downs, conf))

########################################################################
########################################################################
### MATPLOTLIB:

@mpl_check
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
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = data[i, j] / np.min([copy[i, i], copy[j, j]])
        if norm == 'pct':
            data = (data * 100).astype(int)

    fmt = '.2f'
    if all([ (not x%1) for x in data.flatten() ]):
        fmt = 'd' 
        data = data.astype(int)

    if xlabels is None:
        xlabels = [ str(i) for i in range(data.shape[0]) ]
    if ylabels is None:
        ylabels = [ str(i) for i in range(data.shape[1]) ]

    #########

    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(data, cmap = cmap, origin='lower', interpolation='nearest')
  
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
        for idx in range(data.shape[0]):
            for jdx in range(data.shape[1]):
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
