################################################

class odict(dict):

    def __getattr__(self, attr):
        return self[attr]
    
    def __setattr__(self, attr, value):
        self[attr] = value
        
################################################

class adict(dict):

    def __add__(self, value):
        try:
            assert isinstance(value, dict)
        except:
            raise Exception(f"unsupported operand type(s) for +: 'adict' and '{type(value)}'")
        for key, val in value.items():
            if key in self:
                self[key] = self[key] + val
            else:
                self[key] = val
        
    def __radd__(self, value):
        try:
            assert isinstance(value, dict)
        except:
            raise Exception(f"unsupported operand type(s) for +: '{type(value)}' and 'adict'")
        return self.__add__(value)

################################################


