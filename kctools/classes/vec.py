################################################

from kctools.classes.map import Map

################################################

class Vector():
    
    '''
    A Vector() is a class that can
      zero, set, add, and multiply
      with each other (or dicts).
    '''

    def __init__(self, inp = {}):
        self._map = Map()
        self.set(inp, True)
 
    def _keys(self):
        return ( x for x in self._map.keys() )
    
    def _values(self):
        return ( getattr(self, x) for x in self._map.keys() )
    
    def _items(self):
        return ( (x, getattr(self, x)) for x in self._map.keys() )
    
    def __dict__(self):
        return { x: getattr(self, x) for x in self._map.keys() }
    
    ###
    
    def nullify(self):
        '''
        set all attributes to 'zero', by:
        self.attr = 0
        '''
        for attr, value in self._items():
            setattr(self, attr, type(value)())
        return self
    
    ###

    def set(self, inp, add_new = False):
        '''
        set all attributes (found in 'inp'), by:
        self.attr = inp.attr
        '''         
        for attr, value in inp.items():
            if attr in self._keys() or add_new:
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
            if attr in self._keys():
                if type(value) in [set, dict]:
                    value.update(inp)
                else:
                    setattr(self, attr, getattr(self, attr) + value)
            elif add_new:
                self._map.add(attr)
                setattr(self, attr, value)
        return self
    
    ###
    
    def mul(self, inp):
        '''
        set all attributes (found in 'inp', if inp.attr is numeric), by:
        self.attr = self.attr * inp.attr
        '''
        for attr, value in inp.items():
            if type(value) in [int, float, complex]:
                if attr in self._keys():
                    setattr(self, attr, getattr(self, attr) * value)
        return self

################################################

class VecSet(Vector):
    
    '''
    A VecSet() extends Vector():
      other vectors can be loaded
        (or unloaded) from a named
        key, and attributes may be
        reset to the sum of these
        named keys.
    '''

    def __init__(self, inp = {}):
        Vector.__init__(self)
        self._slots = {}
    
    ###
    
    def reset(self):
        '''
        set all attrs to the sum of the values found in _'slots'
        '''
        for attr, value in self._slots():
            if not attr in ['_map', '_slots']:
                setattr(self, attr, type(value)())
        for attr, value in self._slots.items():
            if not attr in ['_map', '_slots']:
                self.add(value)
    
    ###
    
    def load(self, inp):
        '''
        set the value of a 'slot'
        '''
        assert isinstance(inp, dict)
        self._slots.update(inp)
        self.reset()
    
    ###
    
    def unload(self, key):
        '''
        remove a 'slot'
        '''
        if key in self._slots:
            del self._slots[key]
        self.reset()

################################################






