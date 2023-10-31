from .map import Map
from ..kctools import coalesce

################################
class Vector(dict):

    '''
        A Vector is a dictionary that is ordered, by 
          keeping track of its keys in a Map.

        It can also zero itself; or set, add, or multiply
          with other dictionaries (or Vectors).
    '''

    def __init__(self, inp = {}):
        self._map = Map()
        self.set(inp, True)
        self.snap()

    def __repr__(self):
        return str(list(self.items()))
    
    def __getattr__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            return self[attr]
    
    def __setattr__(self, attr, value):
        try:
            object.__getattribute__(self, attr)
            raise Exception(f'"{attr}" is protected, hence cannot be set as a class attribute; use ["{attr}"] instead.')
        except AttributeError:
            self[attr] = value
    
    ###
    
    def keys(self):
        return ( x for x in self._map.keys() )
    
    def values(self):
        return ( self[x] for x in self._map.keys() )
    
    def items(self):
        return ( (x, self[x]) for x in self._map.keys() )
    
    ###
    
    def copy(self):
        return Vector(self.items())
    
    def snap(self):
        self._snap = self.values()
        
    def recall(self):
        return self._snap
    
    ###
    
    def nullify(self):
        '''
            set all attributes to zero
            self.attr = 0
        '''
        for attr, value in self.items():
            self[attr] = type(value)()
        return self
    
    ###
    
    def scale(self, factor):
        '''
            mulltiply all numeric values by a constant factor
            self.attr = f * self.attr
        '''
        for attr, value in self._map.items():
            if type(value) in [int, float, complex]:
                self[attr] = type(attr)(factor * value)
        return self
    
    ###

    def set(self, other, add_new = False):
        '''
            overwrite all attributes (found in 'other')
            self.attr = other.attr
        '''
        for attr, value in other.items():
            if attr in self._map or add_new:
                self._map.add(attr)
                self[attr] = value
        return self
    
    ###
    
    def add(self, other, add_new = False):
        '''
            add all attributes (found in 'other')
            self.attr = self.attr + other.attr
        '''
        for attr, value in other.items():
            if attr in self._map:
                if type(self[attr]) in [set, dict]:
                    self[attr].update(value)
                else:
                    self[attr] = self[attr] + value
            elif add_new:
                    self[attr] = value
        return self
    
    ###
    
    def mul(self, other, add_new = False):
        '''
            multiply all numeric attributes (found in 'other')
            self.attr = self.attr * other.attr
        '''
        for attr, value in other.items():
            if type(value) in [int, float, complex]:
                if attr in self._map:
                    self[attr] = self[attr] * value
                elif add_new:
                    self[attr] = type(value)()
        return self
    
    ###
    
    def apply(self, other, func, add_new = False):
        '''
            set all attributes (found in 'other')
            self.attr = func(self.attr, other.attr)
        '''
        for attr, value in other.items():
            if attr in self._map:
                self[attr] = func(self[attr], value)
            elif add_new:
                self[attr] = func(type(value)(), value)
        return self

    ###
    
    def coalesce(self, other, add_new = False):
        return self.apply(other, coalesce, add_new)
    

################################
class VecSet(Vector):
    
    '''
        A VecSet is a Vector that is also the
          sum of a collection of other Vectors,
          which may be loaded or unloaded with
          a named key.
    '''
    
    def __init__(self):
        Vector.__init__(self)
        self._slots = {}

    ###

    def _reset(self):
        self.nullify()
        for key, value in self._slots.items():
            self.add(value, True)

    ###

    def load(self, key, value):
        assert isinstance(value, dict)
        value = Vector(value)
        self._slots.update({key: value})
        self._reset()

    ###
    
    def unload(self, key):
        if key in self._slots:
            del self._slots[key]
            self._reset()

