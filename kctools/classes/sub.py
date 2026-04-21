from collections import Counter

from ..kctools import endify as _e
from ..kctools import is_np, is_pd, is_tq
from ..kctools import pd_check, npd_check

from . import subconf as conf

if is_np:
    import numpy as np

if is_np and is_pd:
    import pandas as pd
    if is_tq:
        from tqdm import tqdm


class N:
    
    '''
        This class represents one or more
          elements in the power-set 2**L.

        It has methods that are useful
          for exploring the properties
          of collections of notes from 
          an L-tone musical scale.

        So, generally, L will be twelve.

        Development notes:

          The complete state of the object
            is at all times uniquely defined
            by property "._hot":
          
            - it is an np.ndarray
            - it will be either 1-D or 2-D
            - if 1-D it is a 1-hot vector
            - if 2-D it is N 1-hot vectors,
                unique and sorted.

          All methods are written to take 1-D
            arrays as arguments, and wrapped
            to apply along 2-D arrays.
    '''

    ### CONFIG   
    
    _L = 12 # overwrite to warp spacetime
    
    def perfect(self = None, L = _L):
        f = np.power(2, 1/L)
        x = [ 1 ]
        while len(x) < L-1:
            x.append(x[-1]*f)
        y = [ np.abs(i-3/2) for i in x ]
        return y.index(min(y))
    
    _P = perfect()

    ############ BIRTH
    
    @npd_check
    def __init__(self, *inp):
        
        if len(inp) == 1:
            inp = inp[0]
        else:
            if not all([ (type(x) in (bool, int)) for x in inp ]):
                raise TypeError('Incorrect input types.')

        ###    
            
        if isinstance(inp, N):
            self._hot = inp._hot.copy()
            
        else:
            if isinstance(inp, int):
                ndxs = ~ np.where([ int(x) for x in list(f'{int(bin(inp)[2:]):0{N._L}d}') ][-N._L:])[0]

            elif isinstance(inp, (tuple, list, np.ndarray)):
                if len(inp) >= N._L:
                    ndxs = np.where(inp)[0]
                else:
                    ndxs = inp 

            elif isinstance(inp, str):
                try:
                    ndxs = conf.zeitler[inp]
                except KeyError:
                    try:
                        ndxs = conf.names[inp]
                    except KeyError:
                        raise Exception(f'Unknown: "{inp}".')

            else:
                ndxs = []

            self._hot = np.zeros(N._L, dtype=bool)
            self._hot[[ ndx%N._L for ndx in ndxs ]] = 1
    
    ############ ETC

    def __bool__(self):
        return bool(self._hot.sum())
    
    def __len__(self):
        return self._hot.shape[0] if self._hot.ndim > 1 else 1

    def __iter__(self):
        return (N(x) for x in self._hot) if self._hot.ndim > 1 else (N(x) for x in [self._hot])
    
    def __repr__(self):
        return str(self._hot.astype(int))

    def __add__(self, other):
        try:
            other = N(other)
        except Exception:
            raise TypeError
        return N(self).add(other)

    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        try:
            other = N(other)
        except Exception:
            raise TypeError
        return N(self).sub(other)
    
    def __rsub__(self, other):
        try:
            other = N(other)
        except Exception:
            raise TypeError
        return other.sub(N(self))
        
    ############ ORGANIZE
    
    def _crush(self):
        # keeps 2-D arrays unique and sorted
        if self._hot.ndim > 1:
            assert self._hot.ndim == 2
            self._hot = np.unique(self._hot, axis=0)
            if self._hot.shape[0] == 1:
                self._hot = self._hot[0]
            else:
                keys = np.apply_along_axis(lambda x: N(x).idx(), 1, self._hot)
                self._hot = self._hot[np.argsort(keys)]
        return self

    def _array_mapper(method):
        # extends array methods to act on 2-D arrays
        def wrapped(self, *args, **kwargs):
            if self._hot.ndim == 1:
                return method(self, *args, **kwargs)
            else:
                self._hot = [ method(N(x), *args, **kwargs)._hot.tolist() for x in self._hot ]
                self._hot = np.array(self._hot).flatten()
                self._hot = self._hot.reshape((self._hot.shape[0]//N._L, N._L))
                return self._crush()
        return wrapped

    def _tuple_mapper(method):
        # extends bool/int/str/tuple methods to act on 2-D arrays
        def wrapped(self, *args, **kwargs):
            if self._hot.ndim == 1:
                return method(self, *args, **kwargs)
            else:
                return [ method(N(x), *args, **kwargs) for x in self._hot ]
        return wrapped

    ########################################################################
    ######################## NON MUTATING METHODS:
    
    ############ UNIQUE

    @_tuple_mapper
    def hot(self):
        return tuple(int(x) for x in self._hot)

    @_tuple_mapper
    def idx(self):
        return int(''.join(map(str, N(self).compose().hot()))[::-1], 2)
    
    def decimal(self):
        return self.idx() 
    
    @_tuple_mapper
    def notes(self):
        return tuple(np.where(self._hot)[0])
    
    @_tuple_mapper
    def harmonize(self, n_max = 4):
        chords = ( N(self).roll(mdx).chord(n_max).hot() for mdx in range(self.num()) )
        return tuple(zip(self.notes(), chords))
    
    @_tuple_mapper
    def string(self, on = 'x', off = '-'):
        return ''.join([ on if x else off for x in self.hot() ])   
    
    @_tuple_mapper
    def zeitler(self):
        return conf.zeitler.get(self.notes(), '')
    
    @_tuple_mapper
    def name(self):
        name = conf.names.get(self.notes())
        if not name:
            modes = N(self).modes()
            for mode in modes:
                name = conf.names.get(mode.notes())
                if name:
                    if not self._hot[0]:
                        name = f'(rootless) {name}'
                    else:
                        for mdx in range(self.num()):
                            if (self._hot == N(mode).roll(mdx)._hot).all():
                                name = f'({_e(mdx+1)} mode) {name}'
                                break
                    break
        name = name or self.string()
        return name
    
    ############ FORTE CLASSIFICATION
    
    @_tuple_mapper
    def forte_class(self):
        x = self.num()
        clist = sorted( c.notes() for c in N().uv().filt('num', x).sym_forte_prime() )
        y = 1 + clist.index(N(self).sym_forte_prime().notes())
        return (x, y)
    
    @_tuple_mapper
    def forte_string(self):
        x, y = self.forte()
        z = 'Z' if self.zygotic() else ''
        return f'{x}-{z}{y}'  
    
    ############ UNIQUE UP TO ROOT
    
    @_tuple_mapper
    def diff(self):
        return tuple(np.diff(list(self.notes())+[N._L+self.notes()[0]])) if self else ()

    @_tuple_mapper
    def gap_string(self):
        return ''.join(map(str, self.diff()))

    def g_string(self):
        return self.gap_string(self)
    
    ############ UNIQUE UP TO SYMMETRY
    
    @_tuple_mapper
    def num(self):
        return int(self._hot.sum())
    
    @_tuple_mapper
    def perf(self):
        return sum([ bool(self._hot[(n+N._P)%N._L]) for n in self.notes() ])
    
    @_tuple_mapper
    def spin(self):
        return self.chirality() * self.symmetry()

    @_tuple_mapper
    def symmetry(self):
        for m in range(1, 1 + N._L):
            if not N._L % m:
                l = N._L // m
                tiled = np.reshape(self._hot, (m, l))
                if (tiled == tiled[0]).all():
                    sym = m
        return sym

    @_tuple_mapper
    def chirality(self):
        if (N(self).canon()._hot == N(self).rev().canon()._hot).all():
            h = 0
        elif (N(self).canon()._hot == N(self).sym_canon()._hot).all():
            h = -1
        else:
            h = 1
        return h

    def handedness(self):
        return self.chirality()
        
    @_tuple_mapper
    def interval(self):
        out = [0] * (N._L // 2)
        for idx, a in enumerate(self.notes()):
            for jdx, b in enumerate(self.notes()):
                if jdx > idx:
                    odx = (b-a+N._L) % N._L
                    out[odx] += 1
        return tuple(out)
    
    @_tuple_mapper
    def gaps(self):
        return tuple(sorted(Counter(self.diff()).most_common())[::-1])

    @_tuple_mapper
    def runs(self):
        runs = []
        run = 0
        for gap in self.diff():
            if gap == 1:
                run += 1
            elif run:
                runs.append(1+run)
                run = 0
        if run:
            if runs:
                if self.diff()[0] == 1:
                    runs[0] = runs[0] + run
                else:
                    runs.append(1+run)
            else:
                runs.append(1+run)
        return tuple(sorted(Counter(runs).most_common())[::-1])

    @_tuple_mapper
    def energy(self, norm = False, ref = False):
        e, r = 0, 0
        for idx, a in enumerate(self.notes()):
            for jdx, b in enumerate(self.notes()):
                if jdx > idx:
                    e += (b-a)**2
                    e += (N._L-(b-a))**2
                    r += (N._L/self.num()*(jdx-idx))**2
                    r += (N._L-N._L/self.num()*(jdx-idx))**2
        return 1 if ref and norm else r if ref else e/r if norm else e      
    
    ############ CHECK
    
    @_tuple_mapper
    def is_root(self):
        return self._hot[0] == 1
    
    @_tuple_mapper
    def is_prime(self):
        return bool((self._hot == N(self).prime()._hot).all())
    
    @_tuple_mapper
    def is_forte_prime(self):
        return bool((self._hot == N(self).forte_prime()._hot).all())
    
    @_tuple_mapper
    def is_canon(self):
        return bool((self._hot == N(self).canon()._hot).all())

    @_tuple_mapper
    def is_sym_prime(self):
        return bool((self._hot == N(self).sym_prime()._hot).all())

    @_tuple_mapper
    def is_sym_forte_prime(self):
        return bool((self._hot == N(self).sym_forte_prime()._hot).all())
    
    @_tuple_mapper
    def is_sym_canon(self):
        return bool((self._hot == N(self).sym_canon()._hot).all())    
    
    @_tuple_mapper
    def zygotic(self):
        return len(N(self).twins()) > 1
    
    @_tuple_mapper
    def musical(self):
        success = self.is_root()
        success *= (self.num() >= 2)
        success *= (self.num() <= 8)
        run_lim = {2:1, 3:2, 4:2, 5:2}.get(self.num(), 3)
        run_max = self.runs()[0][0] if self.runs() else 0
        success *= run_max <= run_lim
        gap_lim = {2:7, 3:6, 4:5}.get(self.num(), 3)
        gap_max = self.gaps()[0][0] if self.gaps() else N._L
        success *= gap_max <= gap_lim
        return success

   ############ COMPARE
    
    @_tuple_mapper
    def contains(self, other):
        return N(other)._hot == (N(self).compose()._hot * N(other)._hot)

    @_tuple_mapper
    def contained(self, other):
        return N(other).contains(self)
    
    ############ MERGE
    
    @_array_mapper
    def add(self, other):
        return np.minimum(self.compose()._hot + other.compose()._hot, 1)

    @_array_mapper
    def sub(self, other):
        return np.maximum(self.compose()._hot - other.compose()._hot, 0)    
    
    ########################################################################
    ######################## MUTATING METHODS:
        
    ############ SIDEWAYS
    
    @_array_mapper
    def compose(self):
        self._hot = self._hot.astype(bool)
        return self

    @_array_mapper
    def decompose(self):
        self.compose()
        self._hot = self._hot.astype(int)
        self._hot += N(self).complement()._hot
        return self  
    
    ############ ONE-TO-ONE
    
    @_array_mapper
    def inv(self):
        self.compose()
        self._hot = np.invert(self._hot)
        return self

    @_array_mapper
    def anti(self):
        self.roll().inv()
        self._hot[0] = 1
        return self
        
    @_array_mapper
    def reflect(self):
        self._hot = np.roll(np.flip(self._hot), 1)
        return self
    
    def rev(self):
        return self.reflect()
    
    def reverse(self):
        return self.reflect()
    
    @_array_mapper
    def trans(self, ndx = 0):
        ndx = ndx % N._L
        self._hot = np.roll(self._hot, -ndx)
        return self
    
    @_array_mapper
    def sharpen(self, ndx):
        if self.num() and self.num() < N._L:
            mdx = ndx
            while not self._hot[mdx]:
                mdx += 1
            self._hot[mdx] = 0
            mdx += 1
            while self._hot[mdx]:
                mdx += 1
            self._hot[mdx] = 1
        return self

    @_array_mapper
    def flatten(self, ndx):
        if self.num() and self.num() < N._L:
            mdx = ndx
            while not self._hot[mdx]:
                mdx -= 1
            self._hot[mdx] = 0
            mdx -= 1
            while self._hot[mdx]:
                mdx -= 1
            self._hot[mdx] = 1
        return self
    
    ############ MANY-TO-ONE
    
    @_array_mapper
    def roll(self, mdx = 0):
        if self.num():
            mdx = mdx % self.num()
            self.trans(self.notes()[mdx])
        return self
    
    @_array_mapper
    def root(self):
        return self.roll()

    @_array_mapper
    def prime(self):
        self.compose().modes()
        if self._hot.ndim > 1:
            self._hot = self._hot[0, :]
        return self
    
    @_array_mapper
    def canon(self):
        return self.prime().roll(1)

    @_array_mapper
    def forte_prime(self):
        # see sym_forte_prime for forte's actual prime value
        if self:
            ndx = N(self).prime().notes()[-1]
            cands = [ m for m in N(self).modes() if m.notes()[-1] == ndx ]
            cdx = 0
            while len(cands) > 1:
                cdx += 1
                notes = [ c.notes()[cdx] for c in cands ]
                ndx = min(notes)
                cands = [ c for c in cands if c.notes()[cdx] == ndx ]
            self._hot = cands[0]._hot
        return self    
    
    @_array_mapper
    def sym_prime(self):
        self.compose().sym_modes()
        if self._hot.ndim > 1:
            self._hot = self._hot[0, :]
        return self

    @_array_mapper
    def sym_canon(self):
        return self.sym_.prime().roll(1)

    @_array_mapper
    def sym_forte_prime(self):
        # what is actually called forte prime in literature
        if self:
            ndx = N(self).sym_prime().notes()[-1]
            cands = [ m for m in N(self).sym_modes() if m.notes()[-1] == ndx ]
            cdx = 0
            while len(cands) > 1:
                cdx += 1
                notes = [ c.notes()[cdx] for c in cands ]
                ndx = min(notes)
                cands = [ c for c in cands if c.notes()[cdx] == ndx ]
            self._hot = cands[0]._hot
        return self
    
    ############ RESTRICT
    
    @_array_mapper
    def chord(self, n_max = None):
        n_max = n_max or N._L
        n_max = max(n_max, self.num()//2 + self.num()%2)
        notes = self.notes()[::2][:n_max]
        self._hot = N(notes)._hot
        return self

    @_array_mapper
    def complement(self, n_max = None):
        n_max = n_max or N._L
        chord = N(self).chord().notes()
        notes = [ n for n in self.notes() if n not in chord ][:n_max]
        self._hot = N(notes)._hot
        return self  

    ############ ONE-TO-MANY

    @_array_mapper
    def modes(self):
        if self.num():
            self._hot = np.array([ N(self).roll(mdx)._hot for mdx in range(self.num()) ])
        return self._crush()

    @_array_mapper
    def sym_modes(self):
        x = N(self).modes()._hot
        y = N(self).rev().modes()._hot
        if x.ndim > 1:
            assert y.ndim > 1
            self._hot = np.concatenate((x, y), 0)
        else:
            assert y.ndim == 1
            self._hot = np.array([x, y])
        return self._crush()
    
    @_array_mapper
    def twins(self):
        self._hot = N().uv().filt('is_prime').filt('interval', self.interval())._hot
        return self
    
    @_array_mapper
    def parents(self):
        self.compose()
        if not self._hot.sum() == N._L:
            zdxs = list(enumerate(N(self).inv().notes()))
            self._hot = np.tile(self._hot, len(zdxs))
            self._hot = self._hot.reshape((len(zdxs), N._L))
            for i, j in zdxs:
                self._hot[i, j] = 1
        return self._crush()

    @_array_mapper
    def children(self):
        self.compose()
        if self._hot.sum():
            ndxs = list(enumerate(self.notes()))
            self._hot = np.tile(self._hot, len(ndxs))
            self._hot = self._hot.reshape((len(ndxs), N._L))
            for i, j in ndxs:
                self._hot[i, j] = 0
        return self._crush()

    @_array_mapper
    def universe(self):
        self._hot = np.zeros((2**N._L, N._L), dtype=bool)
        for idx in range(2**N._L):
            self._hot[idx, :] = N(idx)._hot
        return self

    def uv(self):
        return self.universe()
    
    ############ BLACK HOLE

    def filt(self, column, value = True):

        test = getattr(N(self), column)()
        if isinstance(test, list):
            test_type = test[0]
        else:
            test_type = test
        
        if not isinstance(test_type, (bool, int, str, tuple)):
            raise Exception(f'Cannot filter by column "{column}"')

        if self._hot.ndim == 2:
            succ = [ x == value for x in test ]
            keep = np.where(succ)[0]
            self._hot = self._hot[keep, :]
            if not self._hot.size:
                self._hot = np.zeros(N._L, dtype=bool)
        else:
            assert self._hot.ndim == 1
            if not test == value:
                self._hot = np.zeros(N._L, dtype=bool)

        return self
    
    ########################################################################
    ######################## WE LOVE TO SPREADSHEET

    def df(self, *args, **kwargs):
        return self.dataframe(*args, **kwargs)

    @pd_check
    def dataframe(self, filt = None, value = True):
        
        # big bang
        if not self.num():
            self.universe()
        
        if filt == True:
            filt = 'musical'
        
        self.filt(filt, value)
            
        cols  = ['hot', 'num', 'perf', 'energy', 'symmetry']
        cols += ['sym_canon', 'runs', 'gaps']
        cols += ['canon', 'chirality']
        cols += ['string', 'name']

        what = tqdm(self) if is_tq and len(self) >= 512 else self
        
        rows = []
        for hot in what:
            row = []
            for col in cols:
                value = getattr(N(hot), col)()
                value = value.hot() if isinstance(value, N) else value 
                row.append(value)
            rows.append(row)
        df = pd.DataFrame(rows, columns=cols)

        df = df.sort_values(
            by = ['num', 'energy', 'symmetry', 'runs', 'gaps', 'chirality'],
            ascending = [True, True, False, False, False, False]
        )
 
        if filt:
            if filt in ['is_root', 'musical']:
                index_col = 'hot'
            if filt in ['is_prime', 'is_canon']:
                index_col = 'canon'
                df = df.drop('hot', axis=1)
            if filt in ['is_sym_prime', 'is_sym_canon']:
                index_col = 'sym_canon'
                df = df.drop('hot', axis=1)
                df = df.drop('canon', axis=1)
            df = df.set_index(index_col, drop=True)
                    
        return df
