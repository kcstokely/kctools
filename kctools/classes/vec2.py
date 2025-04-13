from collections import OrderedDict
from .map import Map
from ..kctools import coalesce

################################
class AttrSpace(dict):

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

################################
class Vector(AttrSpace):

    '''
        A Vector is a dictionary that is ordered, by 
          keeping track of its keys in a Map.

        It can also zero itself; or set, add, or multiply
          with other dictionaries (or Vectors).
    '''

    def __init__(self, inp = {}):
        self._map = Map()
        self.set(inp, True)
        self.snaphot()

    def __repr__(self):
        return str(list(self.items()))
    
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
    
    def recall(self):
        return Vector(self._snap)
    
    def snapshot(self):
        self._snap = self.items()
        return self

    ###
    
    def nullify(self):
        for attr, value in self.items():
            self[attr] = type(value)()
        return self
    
    ###
    
    def __mul__(self, other):

        this = self.copy()      
        types = (None, bool, int, float, complex)
        
        if type(other) in types:
            
            for attr, value in this._map.items():
                if type(value) in types:
                    this[attr] = type(attr)(factor * value) # work with non, bool, etc?
        
        elif isinstance(other, dict):
            
            for attr, value in other.items():
                if type(value) in types:
                    if attr in self._map:
                        this[attr] = this[attr] * type(this[attr])(value)
            
        return this
    
    ###
    
    def __add__(self, other, add_new = False):

        this = self.copy()  
        
        for attr, value in other.items():
            
            if attr in this._map:
                if type(this[attr]) in [set, dict]:
                    this[attr].update(value)
                else:
                    this[attr] = this[attr] + value
            else:
                this._map.add(attr)
                this[attr] = value
                
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
        self._slots = OrderedDict()

    ###

    def _reset(self):
        self.nullify()
        for key, value in self._slots.items():
            self.add(value, True)

    ###

    def load(self, key, value):
        assert isinstance(value, dict) # wut
        value = Vector(value)
        self._slots.update({key: value})
        self._reset()

    ###
    
    def unload(self, key):
        if key in self._slots:
            del self._slots[key]
            self._reset()

