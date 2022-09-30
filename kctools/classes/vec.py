################################################

from .map import Map

################################################

class Vector():
    
    '''
        A Vector() is a class that keeps track of things in a map.
        
        Can zero, set, add, and multiply with each other or dicts.
    '''

    def __init__(self, inp = {}):
        self._map = Map()
        self.set(inp, True)

    def __repr__(self):
        return str(list(self.items()))
        
    def keys(self):
        return ( x for x in self._map.keys() )
    
    def values(self):
        return ( getattr(self, x) for x in self._map.keys() )
    
    def items(self):
        return ( (x, getattr(self, x)) for x in self._map.keys() )
    
    ###
    
    def nullify(self):
        '''
            set all attributes to 'zero', by:
            self.attr = 0
        '''
        for attr in self._map:
            setattr(self, attr, type(getattr(self, attr))())
        return self
    
    ###

    def set(self, inp, add_new = False):
        '''
            set all attributes (found in 'inp'), by:
            self.attr = inp.attr
        '''         
        for attr, value in inp.items():
            if attr in self._map or add_new:
                self._map.add(attr)
                setattr(self, attr, value)
        return self
    
    ###
    
    def add(self, inp, add_new = False):
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
                # unsafe!
                self._map.add(attr)
                setattr(self, attr, value)
        return self
    
    ###
    
    def mul(self, inp, add_new = False):
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
        A VecSet() extends Vector():
          other Vectors can be loaded
            (or unloaded) from a named
            key, and attributes may be
            reset to the sum of these
            named keys.
    '''

    def __init__(self):
        Vector.__init__(self)
        self._slots = []

    ###

    def _reset(self):
        for attr, value in self._map:
            setattr(self, attr, type(value)())
        for slot, vecset in self._slots:
            self.add(vecset, True)

    ###
    
    def load(self, inp):
        self._slots.update(inp)
        self._reset()

    ###
    
    def unload(self, key):
        if key in self._slots:
            del self._slots[key]
            self._reset()

################################################



    
    
    
    


