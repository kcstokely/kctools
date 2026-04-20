from copy import deepcopy

def lockable(method):
    def wrapped(self, *args, **kwargs):
        if hasattr(self, '_lock') and self._lock:
            new = deepcopy(self)
            return method(new, *args, **kwargs)
        else:
            return method(self, *args, **kwargs)
    return wrapped


class Lock:

    def copy(self):
        return deepcopy(self)

    def lock(self, lock = True):
        self._lock = bool(lock)
        return self

    def unlock(self):
        return self.lock(False)
