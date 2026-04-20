import inspect

from ..kctools import is_np, np_check
from .cache import clear_cache, cached

if is_np:
    import numpy as np

####################################

_CARDS = 'akqj09'

_ORDER = [
    'aa', 'a0', 'a9', 'ak', 'aq', 'aj',
    'kk', 'qq', 'jj', 'kq' ,'qj', 'kj',
    'k0', 'j9', 'q0', 'q9', 'j0', 'k9',
    '00', '09', '99'
]

_ROLL = 'kqj'
_KEEP = ['kqj09'] #['kqj', '09']

_WINS = ['aa', 'a0', 'a9']

_UPSETS = [{
      'k9': ['qj'],
      'k0': ['jj'],
      'q0': ['kj'],
      'q9': ['kq']
    },{
      'q0': ['kj'],
      'j0': ['qq'],
      'j9': ['kk'],
      'q9': ['kq']
}]

########################################################################

@np_check
def factory(**kwargs):

    clear_cache()

    kwargs = { k.lower(): v for k, v in kwargs.items() }

    CARDS  = kwargs.get('cards')  or _CARDS
    ORDER  = kwargs.get('order')  or _ORDER
    ROLL   = kwargs.get('roll')   or _ROLL
    KEEP   = kwargs.get('keep')   or _KEEP
    WINS   = kwargs.get('wins')   or _WINS
    UPSETS = kwargs.get('upsets') or _UPSETS
    
    NORM   = len(CARDS)**2
    # put some asserts
    
    ############################################################

    def dist_norm(data):
        to = max( x.exp() for x in data )
        return [ x.scale(to=to) for x in data ]

    def dist_aggr(data, counts = None):
        if counts is None:
            counts = R.MULTS
        assert len(counts) == len(data)
        return sum( (x*y for x, y in zip(counts, data)) , start = WinDist() )

    ####################################

    class WinDist():

        def __init__(self, roll = None, target = None):

            self.norm = 0
            self.wins = 0
            self.outs = np.zeros(len(ORDER), dtype=int)
            self.tars = np.zeros(len(ORDER), dtype=int)

            if roll or target:
                assert roll and target
                roll = R(roll)
                target = R(target)
                self.norm += 1
                self.wins += int(roll > target)
                self.outs[roll.rank] += 1
                self.tars[target.rank] += 1

        def __repr__(self):
            return f"{self.mean():>6f} -- NORM: {self.norm:>7} : WINS: {self.wins:>7} : OUTS: [{', '.join(f'{x:>5}' for x in self.outs)}] : TARS: [{', '.join(f'{x:>5}' for x in self.tars)}]"

        ################################

        def copy(self):
            copy = WinDist()
            copy.norm = self.norm
            copy.wins = self.wins
            copy.outs = self.outs.copy()
            copy.tars = self.tars.copy()
            return copy

        def _dict(self):
            return {
                'norm': self.norm,
                'wins': self.wins,
                'outs': self.outs.tolist(),
                'tars': self.tars.tolist(),
            }

        def roll(self):
            if np.count_nonzero(self.outs) == 1:
                rank = np.where(self.outs)[0][0]
                return R(ORDER[rank])
            return '--'

        def mean(self):
            return self.wins / self.norm

        def validate(self):
            try:
                assert self.wins <= self.norm
                assert self.outs.sum() == self.norm
                assert self.tars.sum() == self.norm
                if self.norm:
                    assert  isinstance(self.exp(), int)
            except:
                print('VALIDATION ERROR:', self._dict())
                raise
            return self

        ################################

        def exp(self):
            if self.norm:
                e = 0
                while e < 8:
                    if self.norm == NORM ** e:
                        return e
                    e += 1
                return np.log(self.norm) / np.log(NORM)

        def scale(self, by = 0, to = None):
            if to is not None:
                by += to - self.exp()
            s = NORM ** by
            s = int(s) if s == int(s) else s
            self *= s
            return self

        ################################

        def _key(self):
            a = self.wins / self.norm
            b = (-self.outs * np.arange(len(ORDER))).sum()
            return (a, b)

        def __gt__(self, other):
            return self._key() > other._key()

        def __lt__(self, other):
            return self._key() < other._key()

        def __ge__(self, other):
            return self._key() >= other._key()

        def __le__(self, other):
            return self._key() <= other._key()

        def __eq__(self, other):
            return self._key() == other._key()

        def __ne__(self, other):
            return self._key() != other._key()

        ################################

        def __add__(self, other):
            copy = self.copy()
            copy.norm += other.norm
            copy.wins += other.wins
            copy.outs += other.outs
            copy.tars += other.tars
            return copy

        def __sub__(self, other):
            copy = self.copy()
            copy.norm += other.norm
            copy.wins += other.norm - other.wins
            copy.outs += other.tars
            copy.tars += other.outs
            return copy

        def __neg__(self):
            return WinDist() - self.copy()

        def __mul__(self, y):
            copy = self.copy()
            copy.norm *= y
            copy.wins *= y
            copy.outs *= y
            copy.tars *= y
            return copy

        def __rmul__(self, y):
            return self.__mul__(y)

    ############################################################

    class R(str):
        
        def __new__(cls, x = 'aa'): # __new__ not __init__ because we're subclassing str
            if isinstance(x, R):
                return x            # IS same object, not a copy
            assert len(x) == 2
            x = x.lower()
            assert all(i in CARDS for i in list(x))
            x = ''.join(sorted(list(x), key = lambda x: CARDS.index(x)))
            assert x in ORDER
            obj = str.__new__(cls, x)
            obj.mult = 1 + (x[0] != x[1])
            obj.rank = ORDER.index(x)
            obj.points = sum( (1+(y[0]!=y[1])) for y in ORDER[:obj.rank] )
            return obj

        ################################

        ###  self  >  target

        def __gt__(self, target):
            target_rank = ORDER.index(str(target))
            if self.rank == target_rank:
                return self.lower() in WINS  ############  (x < b) != (b > x)
            # forwards (backwards?)
            elif str(target) in UPSETS[0]:
                if str(self) in UPSETS[0][str(target)]:
                    return False
            # backwards (forwards?)
            if str(self) in UPSETS[1]:
                if str(target) in UPSETS[1][str(self)]:
                    return True
            return self.rank < target_rank

        def __lt__(self, target):
            return not self.__gt__(target)

        def __ge__(self, target):
            return self.__gt__(target)

        def __le__(self, target):
            return self.__lt__(target)

        def __eq__(self, target):
            return False

        def __ne__(self, target):
            return True

        def __hash__(self):
            return str.__hash__(self) 

        ################################

        @cached
        def options(self, *, highest = False, naive = False, target = None, semi = None):
            options = [ 'keep' ]
            if set(self).intersection(ROLL):
                options.append('roll')
                if not (highest or naive or target):
                    options.append('delay')
            return options

        @cached
        def replacements(self, other):
            orig = set(self).intersection(ROLL)
            if not orig:
                return [self]
            reps = set()
            for group in KEEP:
                reps = set(other).intersection(group)
                if reps:
                    break
            reps = reps or set(other)
            reps = list(set([ R(i+j) for i in orig for j in reps ]))
            return reps

        ################################

        @cached
        def wins(self, *, highest = False, naive = False, target = None, semi = None):
            '''
                given self as final outcome
                calculate wins
            '''
            if highest:
                assert highest in ['rank', 'points']
                wins = WinDist()
                wins.norm = 1
                wins.wins = - getattr(self, highest)
                wins.outs[self.rank] += 1
            elif naive:
                assert naive in ['rank', 'points']
                counts = R.MULTS if naive == 'points' else [1] * len(_ORDER)
                wins = []
                for roll in map(R, ORDER):
                    vals = WinDist()
                    vals += self.wins(target=roll)
                    vals -= roll.wins(target=self)
                    wins.append(vals)
                wins = dist_norm(wins)
                wins = dist_aggr(wins, counts=counts)
            elif target:
                wins = WinDist(self, target)
            elif semi:
                wins = - semi.roll_wins(target=self)  
            else:
                wins = - self.chances(target=self)
            return wins

        ################################

        @cached
        def choose(self, other, *, highest = False, naive = False, target = None, semi = None):
            '''
                starting from self, and given 2nd roll
                calculate wins
            '''
            if highest:
                roll = max(self.replacements(other))
                wins = roll.wins(highest=highest)
                return wins
            else:
                data = []
                for roll in self.replacements(other):
                    wins = roll.wins(naive=naive, target=target, semi=semi)
                    data.append(wins)
                return sorted(data)[-1]

        ################################

        @cached
        def roll_wins(self, *, highest = False, naive = False, target = None, semi = None):
            '''
                starting from self
                weighted sum over all second rolls
            '''
            wins = []
            for roll in map(R, ORDER):
                vals = self.choose(roll, highest=highest, naive=naive, target=target, semi=semi)
                wins.append(vals)
            wins = dist_norm(wins)
            wins = dist_aggr(wins)
            return wins

        @cached
        def option_wins(self, opt, *, highest = False, naive = False, target = None, semi = None):
            '''
                starting from self
                weighted sum over all second rolls, for given option
            '''
            if opt == 'best':
                return self.optimal(highest=highest, naive=naive, target=target, semi=semi)
            elif opt == 'keep':
                wins = self.wins(highest=highest, naive=naive, target=target, semi=semi)
            elif opt == 'roll':
                wins = self.roll_wins(highest=highest, naive=naive, target=target, semi=semi)
            elif opt == 'delay':
                if not semi:
                    wins = - R().chances(highest=highest, naive=naive, target=target, semi=self)
                else:
                    wins = - semi.roll_wins(highest=highest, naive=naive, target=target, semi=self)
            wins.opt = opt
            return wins

        @cached
        def optimal(self, *, highest = False, naive = False, target = None, semi = None):
            '''
                starting from self
                weighted sum over all second rolls, for best option
            '''
            data = []
            for opt in self.options(highest=highest, naive=naive, target=target, semi=semi):
                wins = self.option_wins(opt, highest=highest, naive=naive, target=target, semi=semi)
                data.append(wins)
            return sorted(data)[-1]  

        ################################

        @cached
        def chances(self, *, highest = False, naive = False, target = None, semi = None):
            '''
                independent of self
                weighted sum over all initial rolls
            '''
            wins = []
            for roll in map(R, ORDER):
                vals = roll.optimal(highest=highest, naive=naive, target=target, semi=semi)
                wins.append(vals)
            wins = dist_norm(wins)
            wins = dist_aggr(wins)
            return wins

    ############################################################

    R.CARDS   = CARDS
    R.NORM    = NORM
    R.ROLL    = ROLL
    R.KEEP    = KEEP
    R.WINS    = list(map(R, WINS))
    R.ORDER   = list(map(R, ORDER))
    R.MULTS   = [ x.mult for x in R.ORDER ]
    R.UPSETS  = [ { R(k): [ R(v) for v in vals ] for k, vals in data.items() } for data in UPSETS ]

    R().chances(highest='rank')
    R().chances(highest='points')
    R().chances(naive='rank')
    R().chances(naive='points')
    R().chances()
    
    return R

########################################################################
