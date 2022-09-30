################################################

class odict(dict):
    
    '''
        This is a dictionary, where values can be set
          and accessed like class attributes dict.key

        One could then override dict with: dict=odict

        However, overriding dict doesn't change dicts
          created with { ... } constructors, so it is
          of limited use.
    '''

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

################################################

class adict(dict):

    '''
        This is a dictionary which, when added to another
          dictionary using syntax: dict + dict2, adds the
          values of corresponding keys.
    '''
    
    def __add__(self, value):
        
        if not isinstance(value, dict):
            raise Exception(f"unsupported operand type(s) for +: 'adict' and '{type(value)}'")

        for key, val in value.items():
            try:
                self[key] = self[key] + val
            except KeyError:
                self[key] = val
            except:
                raise
        
    def __radd__(self, value):
        
        if not isinstance(value, dict):
            raise Exception(f"unsupported operand type(s) for +: '{type(value)}' and 'adict'")
        
        return self.__add__(value)

################################################


