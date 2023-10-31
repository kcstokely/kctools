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
                self[key] = self[key] + val
            except KeyError:
                self[key] = val
            except:
                try:
                    self[key].update(self[key])
                except:
                    if not hasattr(self, _flag):
                        raise Exception(f"unsupported operand type(s) for +: 'addict' and '{type(value)}'")
                    elif not self._flag:
                        raise Exception(f"unsupported operand type(s) for +: 'addict' and '{type(value)}'")
                    else:
                        self._flag = 0
                        raise Exception(f"unsupported operand type(s) for +: '{type(value)}' and 'addict'")
        
    def __radd__(self, value):
        
        if not isinstance(value, dict):
            raise Exception(f"unsupported operand type(s) for +: '{type(value)}' and 'addict'")
        
        self._flag = 1    
        return self.__add__(value)
