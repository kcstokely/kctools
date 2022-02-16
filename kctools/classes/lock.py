################################################

'''
    Many classes are developed with methods which
      change state then return the altered class.

    Sometimes, one may wish to call class methods
      without changing state, in order to use the 
      result without mutating the original class.
      
    Child-class methods wrapped with @locker will
      return a copy of self, instead of self, if
      self._lock = True.
      
    Otherwise, a copy() function is provided, to
      invoke this behavior directly.
'''

class Lock():
    
    def __init__(self):
        self._lock = False
    
    def locker(method):
        def wrapped(self, *args, **kwargs):
            if hasattr(self, '_lock') and self._lock:
                new = self.copy()
                return method(new, *args, **kwargs)
            else:
                return method(self, *args, **kwargs)
        return wrapped

    def lock(self, lock = True):
        self._lock = bool(lock)
        return self
    
    def copy(self):
        new = type(self)()
        new.__dict__.update(self.__dict__)
        return new

################################################