################################################

from .map import Map

################################################

class Vector(dict):
    
    '''
        A Vector is an ordered dict (it keeps track of its keys in a Map).

        It can also zero itself, or set, add, or multiply with other dicts.
    '''

    _RESTRICTED_ATTRS = [
        'keys', 'values', 'items', 'snap', 'recall',
        'nullify', 'set', 'add', 'mul', 'apply'
    ]
    
    def __init__(self, inp = {}):
        self._R_ATTRS = Vector._RESTRICTED_ATTRS_
        self._map = Map()
        self.set(inp, True)
        self.snap()

    def __repr__(self):
        return str(list(self.items()))
    
    def __getattr__(self, attr):
        if attr in self._R_ATTRS or attr[0] == '_':
            raise Exception(f'{attr} cannot be accessed as a class attribute.')
        else:
            return self[attr]
        pass
    
    def __setattr__(self, attr, value):
        if attr in self._R_ATTRS or attr[0] == '_':
            raise Exception(f'{attr} cannot be set as a class attribute.')
        else:
            return self[attr] = value
    
    def keys(self):
        return ( x for x in self._map.keys() )
    
    def values(self):
        return ( self[x] for x in self._map.keys() )
    
    def items(self):
        return ( (x, self[x]) for x in self._map.keys() )
    
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
        for attr, value in self.items():
            self[attr] = type(value)()
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
                self[attr] = value
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
                    self[attr] = self[attr] + value
            elif add_new:
                self._map.add(attr)
                self[attr] = type(value)() + value
        return self
    
    ###
    
    def mul(self, inp, add_new = False):
        '''
            set all attributes (found in 'inp'), by:
            self.attr = self.attr * inp.attr
        '''
        for attr, value in inp.items():
            # numeric only
            if type(value) in [int, float, complex]:
                if attr in self._map:
                    self[attr] = self[attr] * value
                elif add_new:
                    self[attr] = type(value)() * value
        return self

    ###
    
    def apply(self, inp, func, add_new = False):
        '''
            set all attributes (found in 'inp'), by:
            self.attr = func(self.attr, inp.attr)
        '''
        for attr, value in inp.items():
            if attr in self._map:
                self[attr] = func(self[attr], value)
            elif add_new:
                self[attr] = func(type(value)(), value)
        return self
    
################################################

class VecSet(Vector):
    
    '''
        A VecSet is a Vector that is also the
          sum of a collection of other Vectors,
          which may be loaded or unloaded with
          a named key.
    '''

    _RESTRICTED_ATTRS = [
        'load', 'unload'
    ]
    
    def __init__(self):
        Vector.__init__(self)
        self._R_ATTRS += VecSet._RESTRICTED_ATTRS_
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
    
    
    
    


