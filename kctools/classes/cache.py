import inspect

_cache = {}


def clear_cache():
    _cache.clear()


def cached(method):

    sig = inspect.signature(method)

    def new_method(*args, **kwargs):

        '''
            key = full.path.name + concat( current value for every parameter in the signature )
        '''
        
        key = [ method.__qualname__ ]

        for pdx, (name, param) in enumerate(sig.parameters.items()):
            if pdx < len(args):
                key.append(args[pdx])
            elif name in kwargs:
                key.append(kwargs[name])
            else:
                key.append(param.default)

        key = '_'.join( repr(k) for k in key )

        try:
            value = _cache[key]
        except KeyError:
            value = method(*args, **kwargs)
            _cache[key] = value

        return value

    new_method.__name__ = method.__name__
    new_method.__signature__ = sig

    return new_method
