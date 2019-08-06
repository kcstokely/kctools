##########################################################################################################################################################

__author__  = 'kcstokely'
__version__ = '0.0.6'

import re
import logging

import numpy as np
import pandas as pd
import datetime as dt

from scipy.stats import beta

##########################################################################################################################################################

def tnow():
    return dt.datetime.strftime(dt.datetime.now(), '%H:%M:%S')

##########################################################################################################################################################

def only_one(thing):
    return sum(map(bool, thing)) == 1

##########################################################################################################################################################

def html_strip(text):
    return re.sub('<[^<]+?>', '', text)

##########################################################################################################################################################

def rep_punc(token):
    return ''.join(list(map(lambda y: y if y not in string.punctuation else ' ', list(token))))

##########################################################################################################################################################

def mround(x, m):
    return int(m * round(float(x)/m))

##########################################################################################################################################################

def split_into_chunks(n, m):
    return [ n//m + (i<n%m) for i in range(m) ]

##########################################################################################################################################################

def kfun(x, p):
    return np.power(np.power(x, 1/p).mean(), p)

def kcos(a, b):
    return np.nan_to_num(np.divide(np.dot(a, b), np.multiply(np.sqrt(np.square(a).sum()), np.sqrt(np.square(b).sum()))), copy=False)

##########################################################################################################################################################

def bin_entropy(true, pred, eps = 0.0000001):
    return -np.sum( true * np.log(pred+eps) + (1-true) * np.log(1-pred+eps) )

def mod_entropy(true, pred, mod = 2., eps = 0.0000001):
    return -np.sum( true * np.log(pred+eps) + (1-true) * (mod * pred) * np.log(1-pred+eps) )

##########################################################################################################################################################

def lower_conf_bound(ups, downs, conf = 0.683):
    return beta.ppf((1.-conf)/2., 1+ups, 1+downs)

def upper_conf_bound(ups, downs, conf = 0.683):
    return beta.ppf(1.-(1.-conf)/2., 1+ups, 1+downs)

def conf_bounds(ups, downs, conf = 0.683):
    return (lower_conf_bound(ups, downs, conf), upper_conf_bound(ups, downs, conf))

##########################################################################################################################################################

def pargsort(arr, n):
    idxs = np.argpartition(arr, n)[:n]
    return idxs[np.argsort(arr[idxs])][:n]

def psort(arr, n):
    return arr[pargsort(arr, n)]

##########################################################################################################################################################

def rchoice(inlist, kwargs = {}):
    return [] if inlist == [] else inlist[np.random.choice(list(range(len(inlist))), **kwargs)]

##########################################################################################################################################################

def flatten(thing):
    if isinstance(thing, list):
        return [ j for i in thing for j in flatten(i) ]
    else:
        return [ thing ]

##########################################################################################################################################################
    
def rowify(inlist, width = 5):
    return [ inlist[i:i+width] for i in range(0, len(inlist), width) ]
    
##########################################################################################################################################################

def gram_getter(items, n, strjoin = False):
    grams = list(zip(*[items[i:] for i in range(n)]))
    if strjoin:
        grams = [ ' '.join(gram) for gram in grams ]
    return grams

##########################################################################################################################################################

def idxs_in_thing(test, thing, idxs = [], already = [], found = False):
    if found or ((thing == test) and (idxs not in already)):
        return idxs, True
    if isinstance(thing, list):
        for i, item in enumerate(thing):
            new_idxs, new_found = idxs_in_thing(test, item, idxs+[i], already)    
            if new_found:
                return new_idxs, True
    return idxs, False

def all_idxs_in_thing(test, thing):
    answers = []
    ans, found = idxs_in_thing(test, thing)
    while(found):
        answers.append(ans)
        ans, found = idxs_in_thing(test, thing, already = answers)
    return answers

##########################################################################################################################################################

def make_dates(num_dates, up_to = 'yesterday', skip = 1, p_range = False):
    if up_to == 'today':
        end_date = dt.datetime.now().date()
    elif up_to == 'yesterday':
        end_date = dt.datetime.now().date() - dt.timedelta(days = 1)
    elif isinstance(up_to, int):
        end_date = dt.datetime.now().date() - dt.timedelta(days = up_to)
    else:
        end_date = dt.datetime.strptime(up_to, '%Y-%m-%d')
    dates = []
    for lag in range(num_dates-1, -1, -1):
        date = end_date - dt.timedelta(days = lag*skip)
        date = dt.datetime.strftime(date, '%Y-%m-%d')
        dates.append(date)
    if p_range:
        starts = []
        for lag in range(num_dates-1, -1, -1):
            date = end_date - dt.timedelta(days = lag*skip+p_range-1)
            date = dt.datetime.strftime(date, '%Y-%m-%d')
            starts.append(date)
        dates = list(zip(starts, dates))
    return dates

##########################################################################################################################################################

# call with __name__

def setup_logger(name, log_file = 'this.log', log_dir = './', mode = 'a', level = logging.INFO):
    assert mode in ['a', 'w']
    logger = logging.getLogger(name)
    logger.setLevel(level)
    f_handler = logging.FileHandler(log_dir + log_file, mode)
    f_handler.setFormatter(logging.Formatter(fmt=f'%(asctime)s - %(levelname)8s: %(message)s', datefmt='%Y-%m-%d - %H:%M:%S'))
    f_handler.setLevel(level)
    logger.addHandler(f_handler)
    return logger

##########################################################################################################################################################

def rename_dup_df_cols(df):
    names = pd.Series(df.columns)
    for dup in df.columns.get_duplicates():
        d_mask = df.columns.get_loc(dup)
        if not isinstance(d_mask, int):
            names[d_mask] = [ dup + '.' + str(ddx) for ddx in range(d_mask.sum()) ]
    df.columns = names
    
##########################################################################################################################################################

class Map(object):

    def __init__(self, items = [], offset = 0):
        assert isinstance(offset, int), 'Offset is not an integer.'
        assert not (offset < 0), 'Offset is not non-negative.'
        self.off = offset
        self.map = dict()
        self.inv = dict()
        self.add(items)
        
    def __len__(self):
        return len(self.map)
    
    def __repr__(self):
        return repr(self.map)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.inv[key]
        else:
            try:
                return self.map[key]
            except:
                self._add_item(key)
                return self.map[key]
        
    def keys(self):
        return self.map.keys()
    
    def values(self):
        return self.map.values()
        
    def _add_item(self, key):
        self.map[key] = len(self)
        self.inv[len(self) + self.off - 1] = key
    
    def _rem_item(self, key):
        if isinstance(key, int):
            del self.map[self.inv[key]]
            for k in range(key, self.off + len(self.map)):
                self.inv[k] = self.inv[k+1]
                self.map[self.inv[k]] = k
            del self.inv[self.off + len(self.map)]
        else:    
            self._rem_item(self.map[key])    
    
    def add(self, thing):    
        if isinstance(thing, list) or isinstance(thing, tuple):
            for item in thing:
                self._add_item(item)
        else:
            self._add_item(thing)

    def rem(self, thing):
        if isinstance(thing, list) or isinstance(thing, tuple):
            for item in thing:
                self._rem_item(item)
        else:
            self._rem_item(thing)

##########################################################################################################################################################