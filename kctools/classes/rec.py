class RecursiveDict(dict):

    def __getitem__(self, key):
        try:
            value = dict.__getitem__(key)
        except KeyError:
            value = None
        if isinstance(value, dict):
            return value
        else:
            for k, v in self.items():
                if isinstance(v, dict):
                    try:
                        x = v.__getitem__(key)
                    except KeyError:
                        x = None
                    if value is None:
                        value = x
                    elif x is not None:
                        value += x
            if value is None:
                raise KeyError
