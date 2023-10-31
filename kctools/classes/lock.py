class Lock():

    '''
        Many classes contain methods which change the
          current state then return the altered class.

        Sometimes, one may wish to call class methods
          without changing state, in order to use the 
          result without mutating the original class.

        Child-class methods wrapped with @locker will
          return a copy of self, instead of self, if
          self._lock = True.

        Otherwise, a copy() function is provided, to
          invoke this behavior directly.

        It is not useful on its own in any way, only
          as something to inherit from.
    '''

    def copy(self):
        new = type(self)()
        new.__dict__.update(self.__dict__)
        return new

    def lock(self, lock = True):
        self._lock = bool(lock)
        return self

    def unlock(self):
        return self.lock(False)

    def locker(method):
        def wrapped(self, *args, **kwargs):
            if hasattr(self, '_lock') and self._lock:
                new = self.copy()
                return method(new, *args, **kwargs)
            else:
                return method(self, *args, **kwargs)
        return wrapped
