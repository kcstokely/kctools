################################################

class Map():

    '''
        This class maps non-integer types to consecutive
          non-negative integers, beginning at 'offset'.

        Generally, these non-integers will be strings.

        Map.keys()   returns the strings, in order.
        Map.values() returns the integers, in order.
        Map.items()  returns the tuples, as expected.

        Map[x] returns the corresponding value to x, ie:
          if x is a string, it returns the mapped integer,
          if x is an integer, it returns the mapped string.

        Slicing is allowed.

        Map.get(x) is equivalent to Map[x], but also allows that
          x may be a (nested) list, in which case it returns the
          expected list of mapped values, and 'None' for missing
          elements.

        Map.add(x) is equivalent to Map.get(x), except missing elements
          will automatically be added, and the new mapped value returned.

        Map.rem(x) is equivalent to Map.get(x), except any found elements
          are then removed from the Map, with integers collapsing downward
          as appropriate to keep them sequential.
    '''
    
    ###############
    
    def __init__(self, inp = [], offset = 0):
        assert isinstance(offset, int), 'Offset is not an integer.'
        assert not (offset < 0), 'Offset is not non-negative.'
        self._off = offset
        self._map = dict()
        self._inv = dict()
        self.add(inp)
        
    def __len__(self):
        return len(self._map)
    
    def __repr__(self):
        return repr(self._map)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start if (key.start == None or isinstance(key.start, int)) else self[key.start]
            stop  = key.stop  if (key.stop  == None or isinstance(key.stop,  int)) else self[key.stop] + 1
            slc   = slice(start, stop, key.step)
            return [ self[x] for x in range(*slc.indices(len(self))) ]
        elif isinstance(key, int):
                if key in self._inv:
                    return self._inv[key]
        else:
            if key in self._map:
                return self._map[key]

    def __iter__(self):
        for x in self.keys():
            yield x
    
    ###############

    def keys(self):
        return ( self._inv[i] for i in self.values() ) 
    
    def values(self):
        return range(self._off, self._off + len(self._map))

    def items(self):
        return zip(self.keys(), self.values())
    
    ###############
    
    def _add_item(self, key):
        assert not isinstance(key, int)
        if not key in self._map:
            idx = len(self) + self._off
            self._map[key] = idx
            self._inv[idx] = key
    
    def _rem_item(self, key):
        if key in self._inv:
            idx = key
            key = self._inv[idx]
            del self._map[key]
            for jdx in range(idx, self._off + len(self._map)):
                self._inv[jdx] = self._inv[jdx+1]
                self._map[self._inv[jdx]] = jdx
            del self._inv[self._off + len(self._map)]
        elif key in self._map:
            self._rem_item(self._map[key])

    ###############

    def get(self, thing):
        if type(thing) in (list, tuple):
            return [ self.get(x) for x in thing ]
        else:
            return self[thing]
    
    def add(self, thing):
        if type(thing) in (list, tuple):
            for item in thing:
                self.add(item)
        else:
            self._add_item(thing)
        return self.get(thing)

    def rem(self, thing):
        thing_two = self.get(thing)
        if type(thing) in (list, tuple):
            for item in self.get(thing_two):
                self.rem(item)
        else:
            self._rem_item(thing)
        return thing_two

################################################













