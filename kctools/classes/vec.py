################################################

from .map import Map

################################################

class Vector():
    
    '''
        A Vector is a class that keeps track of attributes in a Map.
        
        It can zero itself, or set, add, or multiply with other Vectors,
          or dicts.
    '''

    def __init__(self, inp = {}):
        self._map = Map()
        self.set(inp, True)
        self.snap()

    def __repr__(self):
        return str(list(self.items()))
    
    def keys(self):
        return ( x for x in self._map.keys() )
    
    def values(self):
        return ( getattr(self, x) for x in self._map.keys() )
    
    def items(self):
        return ( (x, getattr(self, x)) for x in self._map.keys() )
    
    def snap(self):
        self._snap = self.values()
        
    def recall(self):
        return self._snap
    
    ###
    
    def nullify(self):
        '''
            set all attributes to 'zero', by:
            self.attr = 0
        '''
        for attr, value in self._map.items():
            setattr(self, attr, type(value)())
        return self
    
    ###

    def set(self, inp, add_new = False):
        assert type(inp) in [dict, Vector]
        '''
            set all attributes (found in 'inp'), by:
            self.attr = inp.attr
        '''
        for attr, value in inp.items():
            if attr in self._map or add_new:
                self._map.add(attr)
                setattr(self, attr, value) # oh no what if we overwrite this very function !?
        return self
    
    ###
    
    def add(self, inp, add_new = False):
        assert type(inp) in [dict, Vector]
        '''
            set all attributes (found in 'inp'), by:
            self.attr = self.attr + inp.attr
        '''
        for attr, value in inp.items():
            if attr in self._map:
                if type(value) in [set, dict]:
                    value.update(inp)
                else:
                    setattr(self, attr, getattr(self, attr) + value)
            elif add_new:
                self._map.add(attr)
                setattr(self, attr, value)
        return self
    
    ###
    
    def mul(self, inp, add_new = False):
        assert type(inp) in [dict, Vector]
        '''
            set all attributes (found in 'inp', if inp.attr is numeric), by:
            self.attr = self.attr * inp.attr
        '''
        for attr, value in inp.items():
            if type(value) in [int, float, complex]:
                if attr in self._map:
                    setattr(self, attr, getattr(self, attr) * value)
        return self
    
    ###
    
    def apply(self, inp, func, add_new = False):
        assert type(inp) in [dict, Vector]
        '''
            set all attributes (found in 'inp'), by:
            self.attr = func(self.attr, inp.attr)
        '''
        for attr, value in inp.items():
            if attr in self._map:
                setattr(self, attr, func(getattr(self, attr), value))
            elif add_new:
                setattr(self, attr, func(0, value))
        return self
    
################################################

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
        assert type(value) in [dict, Vector]
        if isinstance(value, dict):
            value = Vector(value)
        self._slots.update({key: value})
        self._reset()

    ###
    
    def unload(self, key):
        if key in self._slots:
            del self._slots[key]
            self._reset()

################################################
    
    
    
    


