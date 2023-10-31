from collections import Counter

from ..kctools import endify as _e
from ..kctools import is_np, is_pd
from ..kctools import pd_check

from .data.config import conf as cf

if is_np:
    import numpy as np

if is_np and is_pd:
    import pandas as pd


class N():
    
    '''
        This class represents one or more
          objects in the power-set 2**L.

        It has methods that are useful
          for exploring the properties
          of collections of notes from 
          an L-tone musical scale.
          
        So generally, L will be twelve.

        All data:
          - kept in self._hot
          - it is an np.ndarray
          - it will be either 1-D or 2-D
          - if 1-D it is a 1-hot vector
          - if 2-D it is N 1-hot vectors,
          -   unique and sorted.

        All methods:
          - rely only on self._hot
          - written to act on 1-D array
          - wrappers extend to act on 2-D
    '''

    ### CONFIG   
    
    _L = 12 # overwrite to warp spacetime
    _conf = cf # overwrite to name things
    _cmap = {v: k for k, v in cf.items()}
    
    ############ BIRTH
    
    def __init__(self, inp = []):

        if not is_np:
            raise Exception('Please install numpy, and maybe pandas.')
        
        if isinstance(inp, N):
            self._hot = inp._hot
        else:
            
            if isinstance(inp, str):
                ndxs = N._conf[inp]
            elif isinstance(inp, int):
                ndxs = np.where([ int(x) for x in list(f'{int(bin(inp)[2:]):0{N._L}d}') ])[0]
            elif len(inp) == N._L:
                ndxs = np.where(inp)[0]
            else:
                ndxs = inp

            self._hot = np.zeros(N._L, dtype=bool)
            self._hot[[ ndx%N._L for ndx in ndxs ]] = 1
    
    ############ WHAT
    
    def __len__(self):
        return self._hot.shape[0] if self._hot.ndim > 1 else 1

    def __iter__(self):
        return (N(x) for x in self._hot) if self._hot.ndim > 1 else (N(x) for x in [self._hot])
    
    def __repr__(self):
        return str(N(self)._hot.astype(int))

    ############ ORGANIZE
    
    def crush(self):
        # keeps 2-D arrays unique and sorted
        if self._hot.ndim > 1:
            assert self._hot.ndim == 2
            self._hot = np.unique(self._hot, axis=0)
            if self._hot.shape[0] == 1:
                self._hot = self._hot[0]
            else:
                keys = np.apply_along_axis(lambda x: N(x).index(), 1, self._hot)
                self._hot = self._hot[np.argsort(keys)]
        return self

    def array_mapper(method):
        # extends array methods to act on 2-D arrays
        def wrapped(self, *args, **kwargs):
            if self._hot.ndim == 1:
                return method(self, *args, **kwargs)
            else:
                self._hot = [ method(N(x), *args, **kwargs)._hot.tolist() for x in self._hot ]
                self._hot = np.array(self._hot).flatten()
                self._hot = self._hot.reshape((self._hot.shape[0]//N._L, N._L))
                return self.crush()
        return wrapped

    def tuple_mapper(method):
        # extends bool/int/tuple methods to act on 2-D arrays
        def wrapped(self, *args, **kwargs):
            if self._hot.ndim == 1:
                return method(self, *args, **kwargs)
            else:
                return [ method(N(x), *args, **kwargs) for x in self._hot ]
        return wrapped

    ############ REVEAL (UNIQUE IDENTIFIERS)

    @tuple_mapper
    def hot(self):
        return tuple(int(x) for x in self._hot)

    @tuple_mapper
    def index(self):
        return int(''.join(map(str, N(self).compose().hot())), 2)
    
    @tuple_mapper
    def notes(self):
        return tuple(np.where(self._hot)[0])

    @tuple_mapper
    def string(self):
        return ''.join([ 'x' if x else '_' for x in self.hot() ])
    
    @tuple_mapper
    def forte(self):
        x = self.num()
        y = 1 + N(self).modes()._hot.tolist().index(N(self).canonical())
        return (x, y)
    
    @tuple_mapper
    def forte_string(self):
        x = self.num()
        y = 1 + N(self).modes()._hot.tolist().index(N(self).canonical())
        return '-'.join(self.forte())
    
    @tuple_mapper
    def diff(self):
        return tuple(np.diff(list(self.notes())+[N._L+self.notes()[0]])) if self.notes() else ()
    
    @tuple_mapper
    def lookup(self):
        name = N._cmap.get(self.notes(), '')
        if not name:
            modes = N(self).modes()
            for mode in modes:
                mode = N(mode)
                name = N._cmap.get(mode.notes())
                if name:
                    if not self._hot[0]:
                        name = f'disp. mode of {name}'
                    else:
                        for mdx in range(self.num()):
                            if (self._hot == N(mode).roll(mdx)._hot).all():
                                name = f'{_e(mdx+1)} mode of {name}'
                                break
                    break
        return name
    
    @tuple_mapper
    def harmonize(self, n_max = 4):
        chords = ( N(self).roll(mdx).chord(n_max).hot() for mdx in range(self.num()) )
        return tuple(zip(self.notes(), chords))
    
    ############ CALCULATE (NON-UNIQUE VALUES)
    
    @tuple_mapper
    def num(self):
        return self._hot.sum()
    
    @tuple_mapper
    def spin(self):
        return self.handedness() * self.symmetry()

    @tuple_mapper
    def symmetry(self):
        for m in range(1, 1 + N._L):
            if not N._L % m:
                l = N._L // m
                tiled = np.reshape(self._hot, (m, l))
                if (tiled == tiled[0]).all():
                    sym = m
        return sym

    @tuple_mapper
    def handedness(self):
        if (N(self).canonical()._hot == N(self).rev().canonical()._hot).all():
            h = 0
        elif (N(self).canonical()._hot == N(self).ext_canonical()._hot).all():
            h = -1
        else:
            h = 1
        return h

    @tuple_mapper
    def is_canon(self):
        return (self._hot == N(self).canonical()._hot).all()
    
    @tuple_mapper
    def is_ext_canon(self):
        return (self._hot == N(self).ext_canonical()._hot).all()
    
    @tuple_mapper
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
    
    @tuple_mapper
    def gaps(self):
        return tuple(sorted(Counter(self.diff()).most_common())[::-1])

    @tuple_mapper
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
    
    ### COMPARE
    
    @tuple_mapper
    def contains(self, other):
        return N(other)._hot == (N(self).compose()._hot * N(other)._hot)

    @tuple_mapper
    def contained(self, other):
        return other.contains(self)

    ############ MUTATE  (PRESERVE)    
    
    @array_mapper
    def compose(self):
        self._hot = self._hot.astype(bool)
        return self

    @array_mapper
    def decompose(self):
        self.compose()
        self._hot = self._hot.astype(int)
        self._hot += N(self).compliment()._hot
        return self  
    
    ############ MUTATE  (PRESERVE / ONE-TO-ONE)
    
    @array_mapper
    def inv(self):
        self.compose()
        self._hot = np.invert(self._hot)
        return self

    @array_mapper
    def rev(self):
        self._hot = np.flip(self._hot)
        return self
    
    @array_mapper
    def trans(self, ndx = 0):
        ndx = ndx % N._L
        self._hot = np.roll(self._hot, -ndx)
        return self
    
    @array_mapper
    def roll(self, mdx = 0):
        if self.num():
            mdx = mdx % self.num()
            self.trans(self.notes()[mdx])
        return self
    
    @array_mapper
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

    @array_mapper
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
    
    ############ MUTATE (PRESERVE / MANY-TO-ONE)
    
    @array_mapper
    def canonical(self):
        self.compose().modes()
        if self._hot.ndim > 1:
            self._hot = self._hot[-1, :]
            self.roll(1)
        return self

    @array_mapper
    def ext_canonical(self):
        self.compose().ext_modes()
        if self._hot.ndim > 1:
            self._hot = self._hot[-1, :]
            self.roll(1)
        return self
    
    ############ MUTATE (RESTRICTT)
    
    @array_mapper
    def chord(self, n_max = None):
        n_max = n_max if n_max else N._L
        n_max = max(n_max, self.num()//2 + self.num()%2)
        notes = self.notes()[::2][:n_max]
        self._hot = N(notes)._hot
        return self

    @array_mapper
    def compliment(self, n_max = None):
        n_max = n_max if n_max else N._L
        chord = N(self).chord().notes()
        notes = [ n for n in self.notes() if n not in chord ][:n_max]
        self._hot = N(notes)._hot
        return self  
    
   ############ MUTATE (EXPAND)

    @array_mapper
    def modes(self):
        if self.num():
            self._hot = np.array([ N(self).roll(mdx)._hot for mdx in range(self.num()) ])
        return self.crush()

    @array_mapper
    def ext_modes(self):
        x = N(self).modes()._hot
        y = N(self).rev().modes()._hot
        if x.ndim > 1:
            assert y.ndim > 1
            self._hot = np.concatenate((x, y), 0)
        else:
            assert y.ndim == 1
            self._hot = np.array([x, y])
        return self.crush()
    
    @array_mapper
    def parents(self):
        self.compose()
        if not self._hot.sum() == N._L:
            zdxs = list(enumerate(N(self).inv().notes()))
            self._hot = np.tile(self._hot, len(zdxs))
            self._hot = self._hot.reshape((len(zdxs), N._L))
            for i, j in zdxs:
                self._hot[i, j] = 1
        return self.crush()

    @array_mapper
    def children(self):
        self.compose()
        if self._hot.sum():
            ndxs = list(enumerate(self.notes()))
            self._hot = np.tile(self._hot, len(ndxs))
            self._hot = self._hot.reshape((len(ndxs), N._L))
            for i, j in ndxs:
                self._hot[i, j] = 0
        return self.crush()
    
    @array_mapper
    def universe(self):
        self._hot = np.zeros((2**N._L, N._L), dtype=bool)
        for idx in range(2**N._L):
            self._hot[idx, :] = N(idx)._hot
        return self

    def uv(self):
        return self.universe()
    
    ### WE LOVE TO SPREADSHEET
    
    @pd_check
    def df(self, filt = None):
        return self.dataframe(filt)
    
    @pd_check
    def dataframe(self, filt = None):

        '''
            Creates a panda of all 2**L sequences, or
              optionally filtered down to a musically
              relevant subset.
        '''
        
        if isinstance(filt, str):
            assert filt in ['', 'hot', 'canon', 'ext_canon']   
            
        cols  = ['hot', 'num', 'energy', 'symmetry']
        cols += ['ext_canonical', 'runs', 'gaps']
        cols += ['canonical', 'handedness', 'string', 'lookup']

        rows = []
        for hot in self:
            row = []
            for col in cols:
                value = getattr(hot, col)()
                value = value.hot() if isinstance(value, N) else value
                row.append(value)
            rows.append(row)
        df = pd.DataFrame(rows, columns=cols)

        if filt:

            # limit to first note present
            df['ok'] = df['hot'].apply(lambda x: x[0])
            df = df.drop(df[df['ok']==0].index).drop(['ok'], axis = 1)

            # limit to num in [3, 8]
            df = df.drop(df[df['num']<3].index)
            df = df.drop(df[df['num']>8].index)

            # limit to no 'large' runs
            df['run_lim'] = df['num'].apply(lambda x: {3:2, 4:2, 5:2}.get(x, 3))
            df['max_run'] = df['runs'].apply(lambda x: x[0][0] if x else 0)
            df = df.drop(df[df['max_run']>df['run_lim']].index)
            df = df.drop(['run_lim', 'max_run'], axis = 1)

            # limit to no 'large' gaps
            df['gap_lim'] = df['num'].apply(lambda x: {3:6, 4:5}.get(x, 3))
            df['max_gap'] = df['gaps'].apply(lambda x: x[0][0] if x else N._L )
            df = df.drop(df[df['max_gap']>df['gap_lim']].index)
            df = df.drop(['gap_lim','max_gap'], axis = 1)

            if filt == 'hot':
                df = df.set_index('hot', drop=True)

            if filt == 'canon':
                df = df[df['canonical']==df['hot']]
                df = df.drop(['hot', 'ext_canonical'], axis=1)
                df = df.set_index('canonical', drop=True)

            if filt == 'ext_canon':
                df = df[df['ext_canonical']==df['hot']]
                df = df.drop(['hot', 'canonical'], axis=1)
                df = df.set_index('ext_canonical', drop=True)

            df = df.sort_values(
                by = ['num', 'energy', 'symmetry', 'runs', 'gaps', 'handedness'],
                ascending = [True, True, False, False, False, False]
            )

        return df
