DEFAULT = True

class addict(dict):

    '''
        This is a dictionary which, when added to another
          dictionary using syntax: dict + dict2, adds the
          values of corresponding keys.
    '''

    def __add__(self, value):
        
        if not isinstance(value, dict):
            raise Exception(f"unsupported operand type(s) for +: 'addict' and '{type(value)}'")

        for key, val in value.items():
            try:
                self[key] += val
            except KeyError:
                if DEFAULT:
                    self[key] = val
                else:
                    raise

    def __radd__(self, value):
        
        if not isinstance(value, dict):
            raise Exception(f"unsupported operand type(s) for +: '{type(value)}' and 'addict'")
        
        return self.__add__(value)
