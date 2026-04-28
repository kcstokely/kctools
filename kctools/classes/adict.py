from copy import deepcopy
from numbers import Number

class AddDict(dict):

    '''
        This is a dictionary which, when added to another
          dictionary using syntax: adict + dict, adds the
          values of corresponding keys.
          
        Similarly, one can multiply, divide, or apply any
          arbitrary function, when values are numeric.

        Can act like a defaultdict, or not.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._default = True

    def default(self, value = True):
        self._default = value
        return self

    ################################
    
    def __add__(self, other):
        
        if not isinstance(other, dict):
            raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

        this = deepcopy(self)
        
        for attr, value in other.items():

            if attr in this:
                if isinstance(this[attr], AddDict):
                    this[attr] = this[attr] + value
                elif isinstance(this[attr], dict):
                    this[attr] = AddDict(this[attr]) + value
                elif isinstance(this[attr], set):
                    this[attr].update(value)
                else:
                    this[attr] = this[attr] + type(this[attr])(value)
            elif self._default:
                this[attr] = value

        return this

    ############

    def __radd__(self, other):
        
        if not isinstance(other, dict):
            raise TypeError(f"unsupported operand type(s) for +: '{type(other)}' and 'type(self)'")
        
        return self.__add__(other)
    
    ################################

    def __iadd__(self, other):

        if not isinstance(other, dict):
            raise TypeError(f"unsupported operand type(s) for +=: '{type(self)}' and '{type(other)}'")
        
        for attr, value in other.items():
            if attr in self:
                if isinstance(self[attr], AddDict):
                    self[attr] = self[attr] + value
                elif isinstance(self[attr], dict):
                    self[attr] = AddDict(self[attr]) + value
                elif isinstance(self[attr], set):
                    self[attr].update(value)
                else:
                    self[attr] = self[attr] + type(self[attr])(value)
            elif self._default:
                self[attr] = value
        
        return self

    ################################
    
    def __mul__(self, other):

        if not isinstance(other, (Number, dict)):
            raise TypeError(f"unsupported operand type(s) for *: '{type(self)}' and 'type(other)'")

        this = deepcopy(self)
            
        if isinstance(other, Number):
        
            if not other:
                for attr, value in this.items(): 
                    this[attr] = type(value)()
            else:
                for attr, value in this.items(): 
                    if isinstance(value, Number):
                        this[attr] = value * type(value)(other)
        
        else:

            for attr, value in other.items():
                if isinstance(value, Number):
                    if attr in this:
                        if isinstance(this[attr], Number):
                            this[attr] = this[attr] * type(this[attr])(value)

        return this

    ############

    def __rmul__(self, other):
        
        if not isinstance(other, (Number, dict)):
            raise TypeError(f"unsupported operand type(s) for *: '{type(other)}' and 'type(self)'")
        
        return self.__mul__(other)

    ################################
    
    def __imul__(self, other):

        if not isinstance(other, (Number, dict)):
            raise TypeError(f"unsupported operand type(s) for *=: '{type(self)}' and '{type(other)}'")
            
        if isinstance(other, Number):
            if not other:
                for attr in list(self.keys()): 
                    if isinstance(self[attr], Number):
                        self[attr] = type(self[attr])()
            else:
                for attr in list(self.keys()): 
                    if isinstance(self[attr], Number):
                        self[attr] = self[attr] * type(self[attr])(other)
        else:
            for attr, value in other.items():
                if isinstance(value, Number):
                    if attr in self:
                        if isinstance(self[attr], Number):
                            self[attr] = self[attr] * type(self[attr])(value)
        
        return self

    ################################
    
    def __sub__(self, other):
        
        if not isinstance(other, (Number, dict)):
            raise TypeError(f"unsupported operand type(s) for -: '{type(self)}' and '{type(other)}'")

        this = deepcopy(self)
            
        if isinstance(other, Number):
            for attr, value in this.items(): 
                if isinstance(value, Number):
                    this[attr] = value - other
                
        else:
            for attr, value in other.items():
                if isinstance(value, Number):
                    if attr in this:
                        if isinstance(this[attr], Number):
                            this[attr] = this[attr] - value
                            
        return this

    ################################
    
    def __isub__(self, other):

        if not isinstance(other, (Number, dict)):
            raise TypeError(f"unsupported operand type(s) for -=: '{type(self)}' and '{type(other)}'")
            
        if isinstance(other, Number):
            for attr in list(self.keys()):
                if isinstance(self[attr], Number):
                    self[attr] = self[attr] - other
        else:
            for attr, value in other.items():
                if isinstance(value, Number):
                    if attr in self:
                        if isinstance(self[attr], Number):
                            self[attr] = self[attr] - value
        
        return self

    ################################
    
    def __rsub__(self, other):
        
        if not isinstance(other, (Number, dict)):
            raise TypeError(f"unsupported operand type(s) for -: '{type(other)}' and '{type(self)}'")
        
        if isinstance(other, Number):
            this = deepcopy(self)
            for attr, value in this.items():
                if isinstance(value, Number):
                    this[attr] = other - value
        else:
            this = deepcopy(other)
            for attr, value in this.items():
                if isinstance(value, Number):
                    if attr in self:
                        if isinstance(self[attr], Number):
                            this[attr] = value - self[attr]
                            
        return this

    ################################
    
    def __truediv__(self, other):
        
        if not isinstance(other, (Number, dict)):
            raise TypeError(f"unsupported operand type(s) for /: '{type(self)}' and 'type(other)'")

        this = deepcopy(self)

        if isinstance(other, Number):
            if not other:
                    raise ZeroDivisionError("division by zero")
            else:
                for attr, value in this.items():
                    if isinstance(value, Number):
                        this[attr] =  value / other
                
        else:
            for attr, value in other.items():
                if isinstance(value, Number):
                    if attr in this:
                        if isinstance(this[attr], Number):
                            this[attr] = this[attr] / value
                            
        return this
    
    ################################

    def __itruediv__(self, other):
        if not isinstance(other, (Number, dict)):
            raise TypeError(f"unsupported operand type(s) for /=: '{type(self)}' and '{type(other)}'")
            
        if isinstance(other, Number):
            if not other:
                raise ZeroDivisionError("division by zero")
            else:
                for attr in list(self.keys()):
                    if isinstance(self[attr], Number):
                        self[attr] = self[attr] / other
        else:
            for attr, value in other.items():
                if isinstance(value, Number):
                    if attr in self:
                        if isinstance(self[attr], Number):
                            self[attr] = self[attr] / value
        
        return self

    ################################
    
    def __floordiv__(self, other):

        if not isinstance(other, (Number, dict)):
            raise TypeError(f"unsupported operand type(s) for /: '{type(self)}' and 'type(other)'")
           
        this = deepcopy(self)
           
        if isinstance(other, Number):
            if not other:
                    raise ZeroDivisionError("division by zero")
            else:
                for attr, value in this.items():
                    if isinstance(value, Number):
                        this[attr] =  value // other
                
        else:
            for attr, value in other.items():
                if isinstance(value, Number):
                    if attr in this:
                        if isinstance(this[attr], Number):
                            this[attr] = this[attr] // type(this[attr])(value)
                            
        return this
    
    ################################
    
    def __ifloordiv__(self, other):
        if not isinstance(other, (Number, dict)):
            raise TypeError(f"unsupported operand type(s) for //=: '{type(self)}' and '{type(other)}'")
            
        if isinstance(other, Number):
            if not other:
                raise ZeroDivisionError("division by zero")
            else:
                for attr in list(self.keys()):
                    if isinstance(self[attr], Number):
                        self[attr] = self[attr] // other
        else:
            for attr, value in other.items():
                if isinstance(value, Number):
                    if attr in self:
                        if isinstance(self[attr], Number):
                            self[attr] = self[attr] // type(self[attr])(value)
        
        return self

    ################################
          
    def __rtruediv__(self, other):

        if not isinstance(other, (Number, dict)):
            raise TypeError(f"unsupported operand type(s) for /: '{type(other)}' and 'type(self)'")
        
        if isinstance(other, Number):
            this = deepcopy(self)
            for attr, value in this.items():
                if isinstance(value, Number):
                    this[attr] =  other / value

        else:
            this = deepcopy(other)
            for attr, value in this.items():
                if isinstance(value, Number):
                    if attr in self:
                        if isinstance(self[attr], Number):
                            this[attr] = value / self[attr]

        return this
    
    ################################
          
    def __rfloordiv__(self, other):

        if not isinstance(other, (Number, dict)):
            raise TypeError(f"unsupported operand type(s) for //: '{type(other)}' and 'type(self)'")
        
        if isinstance(other, Number):
            this = deepcopy(self)
            for attr, value in this.items():
                if isinstance(value, Number):
                    this[attr] =  other // value

        else:
            this = deepcopy(other)
            for attr, value in this.items():
                if isinstance(value, Number):
                    if attr in self:
                        if isinstance(self[attr], Number):
                            this[attr] = value // self[attr]

        return this
        
    ################################
    
    def apply(self, other, func, default = None):

        if not callable(func):
            raise TypeError(f"unsupported argument type for method 'apply': '{type(func)}'")
        
        if not isinstance(other, (Number, dict)):
            raise TypeError(f"unsupported argument type for method 'apply': '{type(other)}'")
        
        if default is None:
            default = self._default
        
        this = deepcopy(self)
        
        if isinstance(other, Number):
            for attr, value in this.items():
                if isinstance(value, Number):
                    this[attr] = func(value, other)
        
        else:
            for attr, value in other.items():
                if isinstance(value, Number):
                    if attr in this:
                        if isinstance(this[attr], Number):
                            this[attr] = func(self[attr], value)
                    elif default:
                        this[attr] = func(type(value)(), value)

        return this
    
    ################################
