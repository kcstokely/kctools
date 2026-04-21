from ..kctools import coalesce
from .adict import AddDict
from .rec import RecursiveDict

################################
class Vector(AddDict, RecursiveDict):

    '''
        AddDict with some maybe convenient methods for chaining operations.
    '''

    def __init__(self, *args, **kwargs):
        AddDict.__init__(self, *args, **kwargs)
        self.snapshot()

    def __repr__(self):
        return str(list(self.items()))
    
    ###
    
    def copy(self):
        return Vector(self.items())
    
    def recall(self):
        self.__dict__.update(self._snap.__dict__)
        return self
    
    def snapshot(self):
        self._snap = self.copy()
        return self

    def snap(self):
        return self.snapshot()
    
    ###
    
    def add(self, other):
        self += other
        return self
        
    def mul(self, other):
        self *= other
        return self
        
    def null(self):
        self *= 0
        return self
    
    def div(self, other):
        self /= other
        return self
    
    def coalesce(self, other):
        self = self.apply(other, coalesce)
        return self
