################################################

import numpy as np
import pandas as pd
from collections import Counter
from kctools import Lock, flatten, endify as _e

from .data.config import conf as cf

################################################

class N(Lock):
    
    '''
        This class represents one or more
          objects in the power-set 2**L.

        It has methods that are useful
          for exploring the properties
          of collections of notes from 
          an L-tone musical scale.

        All data is stored in self._hot:
          - it is an np.ndarray
          - it will either be 1-D or 2-D
          - if 1-D it is a 1-hot vector
          - if 2-D it is like many 1-hots
          - if 2-D it will be unique/sorted

        All methods rely only on self._hot:
          - written to act on 1-D array
          - wrappers extend to act on 2-D
          
        Most methods mutate self:
          - use .copy() to preserve self
    '''

    _L = 12 # overwrite to warp spacetime
    _conf = cf # overwrite to name things
    
    ############ BIRTH
    
    def __init__(self, inp = []):

        Lock.__init__(self)
        
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
    
    def __len__(self):
        return self._hot.shape[0] if self._hot.ndim > 1 else 1

    def __iter__(self):
        return (x for x in self._hot) if self._hot.ndim > 1 else self._hot
    
    def __repr__(self):
        return str(self.copy()._hot.astype(int))

    ############ EXTEND

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
                self._hot = np.array(flatten(self._hot))
                self._hot = self._hot.reshape((self._hot.shape[0]//N._L, N._L))
                return self.crush()
        return wrapped
    
    def tuple_mapper(method):
        # extends tuple methods to act on 2-D arrays
        def wrapped(self, *args, **kwargs):
            if self._hot.ndim == 1:
                return method(self, *args, **kwargs)
            else:
                return [ method(N(x), *args, **kwargs) for x in self._hot ]
        return wrapped
        
    ############ MUTATE (alter)
    
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

    ############ MUTATE (one-way)
    
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
    
    ############ MUTATE (restrict)
    
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
        chord = self.copy().chord().notes()
        notes = [ n for n in self.notes() if n not in chord ][:n_max]
        self._hot = N(notes)._hot
        return self

   ############ MUTATE (extend)

    @array_mapper
    def modes(self):
        if self.num():
            self._hot = np.array([ self.copy().roll(mdx)._hot for mdx in range(self.num()) ])
        return self.crush()

    @array_mapper
    def ext_modes(self):
        x = self.copy().modes()._hot
        y = self.copy().rev().modes()._hot
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
        zdxs = list(enumerate(self.copy().inv().notes()))
        self._hot = np.tile(self._hot, len(zdxs))
        self._hot = self._hot.reshape((len(zdxs), N._L))
        for i, j in zdxs:
            self._hot[i, j] = 1
        return self.crush()

    @array_mapper
    def children(self):
        self.compose()
        ndxs = list(enumerate(self.notes()))
        self._hot = np.tile(self._hot, len(ndxs))
        self._hot = self._hot.reshape((len(ndxs), N._L))
        for i, j in ndxs:
            self._hot[i, j] = 0
        return self.crush()
    
    ############ PERSIST
    
    @array_mapper
    def compose(self):
        self._hot = self._hot.astype(bool)
        return self

    @array_mapper
    def decompose(self):
        self.compose()
        self._hot = self._hot.astype(int)
        self._hot += self.copy().compliment()._hot
        return self

    ############ REVEAL

    @tuple_mapper
    def hot(self):
        return tuple(int(x) for x in self._hot)

    @tuple_mapper
    def index(self):
        return int(''.join(map(str, self.copy().compose().hot())), 2)
    
    @tuple_mapper
    def notes(self):
        return tuple(np.where(self._hot)[0])

    @tuple_mapper
    def diff(self):
        return tuple(np.diff(list(self.notes())+[N._L+self.notes()[0]])) if self.notes() else ()
    
    @tuple_mapper
    def lookup(self):
        cmap = { v: k for k, v in N._conf.items() }
        name = cmap.get(self.notes())
        if not name:
            modes = self.copy().modes()
            if len(modes) > 1:
                for hot in modes._hot:
                    mode = N(hot)
                    name = cmap.get(mode.notes())
                    if name:
                        if not self._hot[0]:
                            name = f'displaced mode of {name}'
                        else:
                            for mdx in range(self.num()):
                                if (self._hot == mode.copy().roll(mdx)._hot).all():
                                    name = f'{_e(mdx+1)} mode of {name}'
                                    break
                        break
        return name if name else ''
    
    ############ SUMMARIZE
    
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
        if (self.copy().canonical()._hot == self.copy().rev().canonical()._hot).all():
            h = 0
        elif (self.copy().canonical()._hot == self.copy().ext_canonical()._hot).all():
            h = -1
        else:
            h = 1
        return h

    @tuple_mapper
    def is_canon(self):
        return (self._hot == self.copy().canonical()._hot).all()
    
    @tuple_mapper
    def is_ext_canon(self):
        return (self._hot == self.copy().ext_canonical()._hot).all()
    
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
    
    ### HARMONIZE
    
    @tuple_mapper
    def harmonize(self, n_max = 4):
        chords = ( self.copy().roll(mdx).chord(n_max).hot() for mdx in range(self.num()) )
        return tuple(zip(self.notes(), chords))
    
    ### UNIVERSE
    
    def dataframe(self, filt = None):
            
        # filt = hot / canon / ext_canon
        
        cols  = ['hot', 'num', 'energy', 'symmetry']
        cols += ['ext_canonical', 'runs', 'gaps']
        cols += ['canonical', 'handedness', 'lookup']

        rows = []
        for idx in range(2**N._L):
            n = N(idx)
            row = []
            for col in cols:
                m = n.copy()
                value = getattr(m, col)()
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
                by=['num', 'energy', 'symmetry', 'runs', 'gaps', 'handedness'],
                ascending = [True, True, False, False, False, False]
            )

        return df

################################################

if __name__ == '__main__':
    
    n = N()
    df = n.dataframe()






